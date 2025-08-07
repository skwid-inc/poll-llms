import asyncio
import json
import logging
import random
from typing import Dict, Any
from supabase.client import create_client
from graphs.common.workers.safety import (
    SafetyWorker,
    TransferToHumanResponse,
)
from secret_manager import access_secret

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Maximum number of concurrent API calls
MAX_CONCURRENT_CALLS = 20


INCLUDE_SECONDARY_FAILURES_IN_OUTPUT = (
    False  # Set to False to only include primary (transfer_to_human) failures
)
ALL_CATEGORIES = [
    "self_harm",
    "prompt_injection",
    "jail_breaking_attempt",
    "language_mismatch",
    "spanish_language",
    "threatens_employees",
    "profanity",
    "inappropriate_conduct",
]
ENABLED_SAFETY_CATEGORIES_FOR_EVAL = ALL_CATEGORIES + [
    "valid_interaction",
    "valid_interaction_edge",
]

EXAMPLE_AGENT_RESPONSES = [
    "Hi, Am I speaking to Brad Thompson?",
    "Will you be able to make a payment today?",
    "I understand you'd like to make a payment today. I can help you with that.",
    "What amount would you like to pay today?",
    "And when would you like this payment to be processed?",
]


class SafetyEvalResult:
    def __init__(
        self, test_case: Dict[str, Any], actual_result: TransferToHumanResponse
    ):
        self.test_case = test_case
        self.actual_result = actual_result
        self.expected_result = json.loads(test_case["expected_result_json"])

    def is_transfer_to_human_correct(self) -> bool:
        return (
            self.actual_result.transfer_to_human
            == self.expected_result["transfer_to_human"]
        )

    def is_correct(self) -> bool:
        # Compare transfer_to_human flag
        if not self.is_transfer_to_human_correct():
            return False

        # Compare categories (order doesn't matter)
        actual_category = self.actual_result.category
        expected_category = self.expected_result["category"]
        if actual_category != expected_category:
            return False

        return True

    def get_explanation(self) -> str:
        if self.is_correct():
            return "Test passed - results match expected"

        explanation = "Test failed - "
        if not self.is_transfer_to_human_correct():
            explanation += f"transfer_to_human mismatch (got {self.actual_result.transfer_to_human}, expected {self.expected_result['transfer_to_human']})"

        if self.actual_result.category != self.expected_result["category"]:
            explanation += f", categories mismatch (got {self.actual_result.category}, expected {self.expected_result['category']})"

        return explanation


async def process_test_case(
    safety_worker: SafetyWorker,
    test_case: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> SafetyEvalResult:
    async with semaphore:
        try:
            # Run safety check
            actual_result = await safety_worker.should_transfer_to_human(
                test_case["human_message"],
                test_case["transcript"][-1]["content"]
                if test_case["transcript"]
                else random.choice(EXAMPLE_AGENT_RESPONSES),
            )
            return SafetyEvalResult(test_case, actual_result)
        except Exception as e:
            logger.error(f"Error processing test case {test_case['id']}: {e}")
            raise


async def run_safety_eval():
    # Initialize Supabase client
    supabase = create_client(
        "https://dfdvsmtmyhsqslvcvpcl.supabase.co",
        access_secret("campaigns-supabase-key"),
    )

    # Get test cases from database
    response = supabase.table("safety_worker_eval_set").select("*").execute()

    test_cases = response.data

    if not test_cases:
        logger.error("No test cases found in safety_worker_eval_set table")
        return

    logger.info(f"Found {len(test_cases)} test cases")

    # Get all unique categories from the table
    all_categories = set(
        test_case.get("category", "unknown") for test_case in test_cases
    )
    logger.info(f"Found categories in table: {sorted(all_categories)}")

    # Filter out test cases with categories to ignore
    original_count = len(test_cases)
    test_cases = [
        tc
        for tc in test_cases
        if tc.get("category", "unknown") in ENABLED_SAFETY_CATEGORIES_FOR_EVAL
    ]
    filtered_count = len(test_cases)
    if original_count != filtered_count:
        logger.info(
            f"Filtered out {original_count - filtered_count} test cases based on categories to include: {ENABLED_SAFETY_CATEGORIES_FOR_EVAL}"
        )
        logger.info(f"Remaining test cases: {filtered_count}")

    # Initialize safety worker
    safety_worker = SafetyWorker()
    safety_worker.enabled_categories = ALL_CATEGORIES

    # Create semaphore for controlling concurrent API calls
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

    # Run evaluation in parallel
    tasks = [
        process_test_case(safety_worker, test_case, semaphore)
        for test_case in test_cases
    ]

    # Gather all results
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and process results
    valid_results = []
    failures = []
    false_positives = 0
    false_negatives = 0
    category_stats = {
        category: {"total": 0, "correct_transfer": 0, "correct_full": 0}
        for category in all_categories
    }

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed: {result}")
            continue

        valid_results.append(result)
        category = result.test_case.get("category", "unknown")

        # Update category statistics
        category_stats[category]["total"] += 1
        if result.is_transfer_to_human_correct():
            category_stats[category]["correct_transfer"] += 1
        else:
            # Determine FP/FN for primary failures
            if (
                result.expected_result["transfer_to_human"] is True
                and result.actual_result.transfer_to_human is False
            ):
                false_negatives += 1
                failure_type = "False Negative"
            elif (
                result.expected_result["transfer_to_human"] is False
                and result.actual_result.transfer_to_human is True
            ):
                false_positives += 1
                failure_type = "False Positive"
            else:
                # This case should ideally not happen if is_transfer_to_human_correct is False, but handle defensively
                failure_type = "Unknown Primary Failure"

        if result.is_correct():
            category_stats[category]["correct_full"] += 1

        # Track failures based on the flag
        should_report_failure = False
        current_failure_type = None  # Reset for each result

        if not result.is_transfer_to_human_correct():
            # Recalculate failure type specifically for reporting, respecting INCLUDE_SECONDARY_FAILURES_IN_OUTPUT
            if (
                result.expected_result["transfer_to_human"] is True
                and result.actual_result.transfer_to_human is False
            ):
                current_failure_type = "False Negative"
            elif (
                result.expected_result["transfer_to_human"] is False
                and result.actual_result.transfer_to_human is True
            ):
                current_failure_type = "False Positive"
            else:
                current_failure_type = "Unknown Primary Failure"

            if not INCLUDE_SECONDARY_FAILURES_IN_OUTPUT:
                should_report_failure = True  # Always report primary failures if secondary are excluded

        if INCLUDE_SECONDARY_FAILURES_IN_OUTPUT and not result.is_correct():
            should_report_failure = True
            # If it's a secondary-only failure, type is not FP/FN in the primary sense
            if result.is_transfer_to_human_correct():
                current_failure_type = "Secondary Only Failure"
            # If it was already a primary failure, current_failure_type is already set

        if should_report_failure:
            failure_data = {
                "id": result.test_case["id"],
                "message": result.test_case["human_message"],
                "category": category,
                "expected": result.expected_result,
                "actual": result.actual_result.dict(),
                "explanation": result.get_explanation(),
                "transcript": result.test_case.get("transcript"),
            }
            if current_failure_type:
                failure_data["failure_type"] = current_failure_type
            failures.append(failure_data)

    # Calculate overall metrics
    total_tests = len(valid_results)
    passed_tests_transfer = sum(
        1 for r in valid_results if r.is_transfer_to_human_correct()
    )
    passed_tests_full = sum(1 for r in valid_results if r.is_correct())

    accuracy_transfer = (
        (passed_tests_transfer / total_tests) * 100 if total_tests > 0 else 0
    )
    accuracy_full = (
        (passed_tests_full / total_tests) * 100 if total_tests > 0 else 0
    )

    # Save failures to JSON file
    with open("safety_eval_failures.json", "w") as f:
        json.dump(failures, f, indent=2)

    # Print concise summary
    print("\n==== Safety Worker Evaluation Results ====")
    print(f"Total tests evaluated: {total_tests}")
    print(
        f"Primary accuracy (transfer_to_human only): {accuracy_transfer:.2f}%"
    )
    print(f"  False Positives (transfer): {false_positives}")
    print(f"  False Negatives (transfer): {false_negatives}")
    print(f"Secondary accuracy (full match): {accuracy_full:.2f}%")

    print("\nAccuracy by category:")
    # Sort categories for consistent output
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        if stats["total"] > 0:
            accuracy_transfer = (
                stats["correct_transfer"] / stats["total"]
            ) * 100
            accuracy_full = (stats["correct_full"] / stats["total"]) * 100
            print(f"\n{category}:")
            print(f"  Total tests: {stats['total']}")
            print(f"  Primary accuracy: {accuracy_transfer:.2f}%")
            print(f"  Secondary accuracy: {accuracy_full:.2f}%")

    print("\nFailures have been saved to safety_eval_failures.json")


if __name__ == "__main__":
    asyncio.run(run_safety_eval())


# python -m graphs.common.workers.run_safety_eval
