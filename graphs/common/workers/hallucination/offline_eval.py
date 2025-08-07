"File to run offline hallucination analysis on prior conversations retrieved from logs in supabase"

import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any, Optional
from supabase.client import create_client
from dotenv import load_dotenv
import asyncio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from langchain_core.messages import AIMessage
from graphs.common.workers.hallucination.offline_eval_utils import (
    serialize_message,
    compute_metrics,
    OfflineHallucinationAnalysis,
    raw_to_json,
)
from graphs.common.workers.utils import (
    standardize_message,
    openai_chat_to_raw_langchain_messages,
)

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

root_dir = project_root
env_path = root_dir / ".env.local"
if not env_path.exists():
    raise FileNotFoundError(
        f"Environment file not found at {env_path}. "
        "Please ensure .env.local exists in the root directory with required keys: "
        "supabase_key and OPENAI_API_KEY"
    )

os.chdir(root_dir)
load_dotenv(env_path)

from graphs.common.workers.hallucination import (
    get_detector,
)
from graphs.common.workers.utils import format_conversation_history
from graphs.common.workers.hallucination.model_types import (
    HallucinationOutput,
    DetectionLevel,
)
from graphs.common.workers.utils import build_conversation_history

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
supabase_key = os.getenv("supabase_key", "")
if not supabase_key:
    raise ValueError("supabase_key environment variable is not set")

supabase_client = create_client(SUPABASE_URL, supabase_key)


async def retrieve_turns(
    supabase_function: str = "get_westlake_call_turns_custom", limit: int = 50
) -> List[Dict[str, Any]]:
    print(f"Executing stored function: {supabase_function}")
    response = supabase_client.rpc(supabase_function, {"result_limit": limit}).execute()

    turns = response.data
    logger.info(f"Found {len(turns)} turns from stored function")
    return turns


async def analyze_turns_with_detector(
    turns: List[Dict[str, Any]], detector_name: str
) -> List[OfflineHallucinationAnalysis]:
    if not turns:
        logger.info("No turns provided for analysis")
        return []

    detector_func = get_detector(detector_name)
    logger.info(f"Analyzing {len(turns)} turns using detector: {detector_name}")

    bsz = 100
    jobs = min(bsz, len(turns))
    semaphore = asyncio.Semaphore(jobs)

    results = []
    batch_size = bsz

    async def process_row(row):
        async with semaphore:
            return await analyze_row_with_detector(row, detector_func, detector_name)

    pbar = tqdm(total=len(turns), desc=f"Analyzing with {detector_name}", unit="turn")

    async def process_with_progress(row):
        result = await process_row(row)
        pbar.update(1)
        return result

    for i in range(0, len(turns), batch_size):
        batch_turns = turns[i : i + batch_size]
        tasks = [process_with_progress(row) for row in batch_turns]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that were returned
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(
                    f"Exception in batch {i // batch_size}, task {j}: {result}"
                )
                # Re-raise the exception to stop execution for debugging
                raise result
            results.append(result)

    pbar.close()
    return results


async def analyze_row_with_detector(
    row: Dict[str, Any], detector_func, detector_name: str
) -> OfflineHallucinationAnalysis:
    # 1) Build the conversation_history BaseMessages - NEVER MODIFY THIS
    if row.get("conversation_history"):
        raw = row["conversation_history"]
        dicts = raw_to_json(raw)
        conv_history = [standardize_message(msg) for msg in dicts]
    else:
        agent_dicts = openai_chat_to_raw_langchain_messages(
            raw_to_json(row.get("messages", []))
        )  # these messages follow a diferent format

        agent_messages = [standardize_message(msg) for msg in agent_dicts]
        full_dicts = raw_to_json(row.get("full_message_history", []))
        full_messages = [standardize_message(msg) for msg in full_dicts]
        conv_history = build_conversation_history(full_messages, agent_messages)

    # 2) Build the latest messages for detector analysis
    text_out = row.get("agent_text_output") or ""
    tool_out = row.get("agent_tool_output") or []
    # Handle NaN/None values for tool_out
    if tool_out is None or (isinstance(tool_out, float) and pd.isna(tool_out)):
        tool_out = []
    # If tool_out is string, parse it
    if isinstance(tool_out, str):
        tool_out = raw_to_json(tool_out)
    assert isinstance(tool_out, list), f"tool_out is not a list: {tool_out}"

    # Create AI message with both text content and tool calls
    latest_ai = None
    if text_out or tool_out:
        # Convert tool calls to the format expected by AIMessage
        kwargs = {"content": text_out or ""}  # Default to empty string if None
        tool_calls = []
        if tool_out:
            for tool_call in tool_out:
                if isinstance(tool_call, dict):
                    arguments = tool_call.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except (json.JSONDecodeError, ValueError):
                            arguments = {}
                    elif not isinstance(arguments, dict):
                        arguments = {}

                    tool_calls.append(
                        {
                            "id": tool_call.get("id", ""),
                            "name": tool_call.get("name", ""),
                            "args": arguments,
                        }
                    )

        if tool_calls:
            kwargs["tool_calls"] = tool_calls

        latest_ai = AIMessage(**kwargs)

    # 3) Call the detector with the conversation history and latest AI message
    result: HallucinationOutput = await detector_func(
        conversation_history=conv_history,
        latest_ai_message=latest_ai,
        agent_name=row.get("agent_name", ""),
    )

    # 4) Serialize the ORIGINAL conversation_history back to list-of-dicts (NEVER MODIFIED)
    conv_dicts = [serialize_message(m) for m in conv_history]

    # 5) Coerce the other fields safely
    def safe_str(val):
        if val is None or (isinstance(val, float) and pd.isna(val)) or val == "":
            return ""
        return str(val)

    analysis = OfflineHallucinationAnalysis(
        call_id=str(row.get("call_id", "")),
        turn_id=int(row.get("turn_id", 0)),
        agent_name=row.get("agent_name", ""),
        human_input=safe_str(row.get("human_input")),
        agent_text_output=safe_str(row.get("agent_text_output")),
        judge_label="positive"
        if result.detection_level == DetectionLevel.DETECTED
        else "negative",
        true_label=safe_str(row.get("true_label")),
        comments=safe_str(row.get("comments")),
        explanation=result.explanation or "",
        conversation_history=conv_dicts,  # ORIGINAL, NEVER MODIFIED
    )

    return analysis


async def save_analysis_summary(
    analyses: List[OfflineHallucinationAnalysis],
    detector_name: str,
    out_file: str,
    positive_results_file: Optional[str] = None,
):
    if not analyses:
        logger.info("No analyses to summarize")
        return

    detection_levels = {}
    call_ids_by_detection_level = {}
    all_unique_call_ids = set()

    for analysis in analyses:
        level = analysis.judge_label
        detection_levels[level] = detection_levels.get(level, 0) + 1

        if level not in call_ids_by_detection_level:
            call_ids_by_detection_level[level] = set()
        call_ids_by_detection_level[level].add(analysis.call_id)

        all_unique_call_ids.add(analysis.call_id)

    print(f"\n=== Results for {detector_name} ===")
    print(f"Analyzed {len(analyses)} conversation turns")
    print(f"Detection level breakdown: {detection_levels}")
    print(f"Total unique calls analyzed: {len(all_unique_call_ids)}")

    print("\nUnique calls by detection level:")
    for level in sorted(detection_levels.keys()):
        unique_calls = len(call_ids_by_detection_level.get(level, set()))
        print(f"  {level}: {unique_calls} unique calls")

    metrics = compute_metrics(analyses)
    if metrics["valid_count"] > 0:
        print(
            f"\n=== Performance Metrics (based on {metrics['valid_count']} rows with valid true_label) ==="
        )
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1 Score: {metrics['f1']:.3f}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(
            f"Confusion Matrix: TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}"
        )
    else:
        print(f"\nNo valid true_label values found for metrics computation")

    print(f"Results saved to: {out_file}")
    if positive_results_file:
        print(f"Positive detections saved to: {positive_results_file}")
    print("=" * 50)


async def read_validated_csv(csv_path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    return df.to_dict("records")


async def main():
    parser = argparse.ArgumentParser(description="Run offline hallucination analysis")
    parser.add_argument(
        "-l", "--limit", type=int, help="how many turns to retrieve from supabase"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../dump/math_validated.csv",
        help="If you're taking input from csv file",
    )
    parser.add_argument("--output", type=str, help="Custom output filename")
    args = parser.parse_args()

    detector_name = "math_detector"  # "ghost_payment_detector" # "payment_validation_detector" # math_detector

    if args.limit is not None:
        print(f"Retrieving {args.limit} turns from logs...")
        function_name = "westlake_collections_en_llm_messages"  # westlake_collections_en_llm_messages # westlake_collections_en_tool_messages
        turns = await retrieve_turns(function_name, args.limit)
        if not turns:
            print("No turns found from stored function")
            return

        if args.output:
            output_path = Path(args.output)
            base_filename = f"../dump/{output_path.name}"
        else:
            base_filename = f"../dump/{detector_name}"
    else:
        print(f"Reading from {args.input}...")
        turns = await read_validated_csv(args.input)
        if not turns:
            print(f"No data found in {args.input}")
            return

        if args.output:
            output_path = Path(args.output)
            base_filename = f"../dump/{output_path.name}"
        else:
            csv_path = Path(args.input)
            base_filename = f"../dump/{csv_path.stem}_analyzed_{detector_name}"

    results = await analyze_turns_with_detector(
        turns=turns, detector_name=detector_name
    )

    output_path = Path(base_filename)
    is_full_filename = output_path.suffix.lower() in [".csv", ".json", ".xlsx"]

    if is_full_filename:
        df_all = pd.DataFrame([result.model_dump() for result in results])
        df_all.to_csv(base_filename, index=False)
        print(f"All results saved to: {base_filename}")

        positive_results = [
            result for result in results if result.judge_label == "positive"
        ]
        if positive_results:
            print(
                f"Found {len(positive_results)} positive detections out of {len(results)} total"
            )
        else:
            print("No positive detections found")
    else:
        if args.limit is None:
            df_all = pd.DataFrame([result.model_dump() for result in results])
            all_results_file = f"{base_filename}.csv"
            df_all.to_csv(all_results_file, index=False)
            print(f"Results saved to: {all_results_file}")

            positive_results = [
                result for result in results if result.judge_label == "positive"
            ]
            if positive_results:
                print(
                    f"Found {len(positive_results)} positive detections out of {len(results)} total"
                )
            else:
                print("No positive detections found")
        else:
            df_all = pd.DataFrame([result.model_dump() for result in results])
            all_results_file = f"{base_filename}_all.csv"
            df_all.to_csv(all_results_file, index=False)
            print(f"All results saved to: {all_results_file}")

            positive_results = [
                result for result in results if result.judge_label == "positive"
            ]
            if positive_results:
                df_positive = pd.DataFrame(
                    [result.model_dump() for result in positive_results]
                )
                positive_results_file = f"{base_filename}_positive_only.csv"
                df_positive.to_csv(positive_results_file, index=False)
                print(f"Positive detections saved to: {positive_results_file}")
                print(
                    f"Found {len(positive_results)} positive detections out of {len(results)} total"
                )
            else:
                print("No positive detections found")

    await save_analysis_summary(
        results,
        detector_name,
        base_filename if is_full_filename else all_results_file,
        positive_results_file if args.limit is not None and positive_results else None,
    )


if __name__ == "__main__":
    asyncio.run(main())
