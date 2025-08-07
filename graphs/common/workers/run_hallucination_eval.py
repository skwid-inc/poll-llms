import asyncio
import logging
import pandas as pd
from typing import Dict, Any, List
from supabase.client import create_client
from secret_manager import access_secret
from graphs.common.workers.hallucination.offline_eval import analyze_row_with_detector
from graphs.common.workers.hallucination.utils import get_detector
from graphs.common.workers.hallucination.offline_eval_utils import (
    OfflineHallucinationAnalysis,
    compute_metrics,
)
import os
from dotenv import load_dotenv

load_dotenv(".env.local")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_CONCURRENT_CALLS = 50


class HallucinationEvalResult:
    def __init__(
        self, test_case: Dict[str, Any], analysis: OfflineHallucinationAnalysis
    ):
        self.test_case = test_case
        self.analysis = analysis
        self.flagged = analysis.judge_label == "positive"
        self.true_hallucination = test_case.get("true_label", "").lower() == "positive"

    def is_correct(self) -> bool:
        return self.flagged == self.true_hallucination

    def get_explanation(self) -> str:
        return self.analysis.explanation or ""


async def process_test_case(
    detector_func,
    detector_name: str,
    test_case: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> HallucinationEvalResult:
    async with semaphore:
        try:
            analysis = await analyze_row_with_detector(
                test_case, detector_func, detector_name
            )
            return HallucinationEvalResult(test_case, analysis)
        except Exception as e:
            logger.error(
                f"Error processing test case {test_case.get('id', 'unknown')}: {e}"
            )
            raise


async def run_hallucination_eval():
    client_name = os.getenv("CLIENT_NAME")
    supabase_url = os.getenv("SUPABASE_URL")
    if not client_name:
        logger.error("CLIENT_NAME unavailable")
        return
    if not supabase_url:
        logger.error("SUPABASE_URL unavailable")
        return

    logger.info(f"Running hallucination evaluation for client: {client_name}")

    supabase = create_client(
        supabase_url,
        access_secret("campaigns-supabase-key"),
    )

    response = (
        supabase.table("hallucination_worker_eval_set")
        .select("*")
        .eq("client", client_name)
        .execute()
    )
    test_cases = response.data

    if not test_cases:
        logger.error(
            f"No test cases found in hallucination_worker_eval_set table for {client_name}"
        )
        return

    logger.info(f"Found {len(test_cases)} test cases for {client_name}")

    detector_groups = {}
    for test_case in test_cases:
        detector = test_case.get("detector", None)
        if not detector:  # we test row for a specific detector
            continue
        if detector not in detector_groups:
            detector_groups[detector] = []
        detector_groups[detector].append(test_case)

    all_failures = []

    for detector_name, test_cases in detector_groups.items():
        logger.info(
            f"Evaluating detector: {detector_name} with {len(test_cases)} test cases"
        )
        try:
            detector_func = get_detector(detector_name)
        except ValueError as e:
            logger.warning(f"Unable to load '{detector_name}', skipping : {e}")
            continue
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

        tasks = [
            process_test_case(detector_func, detector_name, test_case, semaphore)
            for test_case in test_cases
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results: List[HallucinationEvalResult] = []
        failures = []

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed: {result}")
                continue

            valid_result: HallucinationEvalResult = result  # type: ignore
            valid_results.append(valid_result)

            if valid_result.is_correct():
                continue
            else:
                failure_data = dict(valid_result.test_case)
                failure_data.update(
                    {
                        "judge_label": "positive"
                        if valid_result.flagged
                        else "negative",
                        "explanation": valid_result.get_explanation(),
                    }
                )
                failures.append(failure_data)

        all_failures.extend(failures)

        if valid_results:
            analyses = [r.analysis for r in valid_results]
            metrics = compute_metrics(analyses)

            print(f"\n==== Results for {detector_name} ====")
            print(f"Total tests: {metrics['valid_count']}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"F1 Score: {metrics['f1']:.3f}")
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(
                f"Confusion Matrix: TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}"
            )

            if metrics["f1"] <= 0.92:
                print(
                    f"⚠️  WARNING: Potential regression detected! F1 score ({metrics['f1']:.3f}) is <= 92%"
                )

    if all_failures:
        df = pd.DataFrame(all_failures)
        dump_dir = "../dump"
        os.makedirs(dump_dir, exist_ok=True)
        output_path = os.path.join(dump_dir, "hallucination_eval_failures.csv")
        df.to_csv(output_path, index=False)
        absolute_path = os.path.abspath(output_path)
        print(f"\nAll failed rows saved to: {absolute_path}")
    else:
        print(f"\nNo failures to save")


if __name__ == "__main__":
    asyncio.run(run_hallucination_eval())

# python -m graphs.common.workers.run_hallucination_eval
