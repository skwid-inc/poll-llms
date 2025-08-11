#!/usr/bin/env python3
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)


sys.path.append(parent)

import argparse
import json
import os
import random
import string
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tabulate import tabulate
except Exception:
    tabulate = None  # type: ignore

# Optional imports; handle gracefully if not installed
try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from groq import Groq
except Exception as e:  # pragma: no cover
    Groq = None  # type: ignore
    
try:
    # Use repo's secret manager for consistent key retrieval
    from secret_manager import access_secret
except Exception:
    print("couldn't load secret")
    access_secret = None  # type: ignore

try:
    from otel import report_gauge, report_metrics, inc_counter, init_otel
except Exception as e:
    print(f"couldn't load otel: {e}")
    report_gauge = None  # type: ignore
    report_metrics = None  # type: ignore
    inc_counter = None  # type: ignore
    init_otel = None  # type: ignore


@dataclass
class TestCase:
    file: str
    system_prompt: str
    user_message: str


@dataclass
class RunResult:
    ttft_s: float  # Time to first token
    ttfs_s: Optional[float]  # Time to first sentence (when first sentence ends)
    total_latency_s: float  # Total completion time
    output_chars: int
    ok: bool
    error: Optional[str] = None
    draft_latency_s: Optional[float] = None
    output: Optional[str] = None
    draft_output: Optional[str] = None


@dataclass
class CaseAggregate:
    groq: List[RunResult]
    openai_baseline: List[RunResult]
    openai_spec: List[RunResult]


def read_test_cases(evals_dir: Path) -> List[TestCase]:
    cases: List[TestCase] = []
    for fp in sorted(evals_dir.glob("*.txt")):
        text = fp.read_text(encoding="utf-8", errors="ignore")
        # Split on sentinel lines 'assistant' and 'user' (on their own line)
        lines = [ln.strip("\r") for ln in text.splitlines()]

        # Find the first 'assistant' marker
        try:
            idx_assistant = next(i for i, ln in enumerate(lines) if ln.strip().lower() == "assistant")
        except StopIteration:
            # No interaction found; skip
            continue

        system_lines = lines[:idx_assistant]

        # Find the subsequent 'user' marker
        try:
            idx_user = next(
                i for i in range(idx_assistant + 1, len(lines)) if lines[i].strip().lower() == "user"
            )
        except StopIteration:
            continue

        # Grab user content until next role marker or EOF
        next_marker_idx = None
        for j in range(idx_user + 1, len(lines)):
            if lines[j].strip().lower() in {"assistant", "tool", "system"}:
                next_marker_idx = j
                break
        if next_marker_idx is None:
            next_marker_idx = len(lines)

        user_chunk = "\n".join(ln for ln in lines[idx_user + 1 : next_marker_idx]).strip()
        system_chunk = "\n".join(ln for ln in system_lines).strip()

        if not system_chunk or not user_chunk:
            continue

        cases.append(TestCase(file=fp.name, system_prompt=system_chunk, user_message=user_chunk))

    return cases


# Embedded cases module (preferred)
try:
    from bench_cases import DEFAULT_TEST_CASES as EMBEDDED_CASES  # Remove the 'scripts.' prefix
except Exception:
    EMBEDDED_CASES = []  # type: ignore


def warmup_openai(client: "OpenAI", model: str, count: int = 2, use_spec: bool = False):
    for _ in range(count):
        uid = str(uuid.uuid4())
        messages = [
            {"role": "system", "content": add_cache_breaker("You are a helpful assistant.")},
            {"role": "user", "content": f"Warm-up ping {uid}. Reply with a single word."},
        ]
        extra_body = None
        if use_spec:
            # Warm predicted outputs path with small stub draft
            extra_body = {"prediction": {"type": "content", "content": "ok"}}
        try:
            client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=16,
                temperature=0,
                extra_body=extra_body,
            )
        except Exception:
            # Warm-up is best-effort
            pass


def warmup_groq(client: "Groq", model: str, count: int = 2):
    for _ in range(count):
        uid = str(uuid.uuid4())
        try:
            client.chat.completions.create(
                model=model,
                service_tier="flex",
                messages=[
                    {"role": "user", "content": f"Warm-up ping {uid}. Say 'ok'."},
                ],
                max_tokens=8,
                temperature=0,
            )
        except Exception:
            pass


def run_groq(client: "Groq", model: str, system_prompt: str, user_message: str, timeout_s: float) -> RunResult:
    t0 = time.perf_counter()
    try:
        messages = [
            {"role": "system", "content": add_cache_breaker(system_prompt)},
            {"role": "user", "content": user_message},
        ]
        # Groq SDK does not expose a per-request timeout in create(); rely on env/network settings
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        delta = time.perf_counter() - t0
        content = resp.choices[0].message.content if resp and resp.choices else ""
        # For non-streaming, TTFS is the same as total latency since we get everything at once
        ttfs_time = delta if content and detect_first_sentence_end(content) else None
        return RunResult(ttft_s=delta, ttfs_s=ttfs_time, total_latency_s=delta, output_chars=len(content or ""), ok=True, output=content)
    except Exception as e:
        delta = time.perf_counter() - t0
        return RunResult(ttft_s=delta, ttfs_s=None, total_latency_s=delta, output_chars=0, ok=False, error=str(e))


def run_openai(
    client: "OpenAI",
    model: str,
    system_prompt: str,
    user_message: str,
    timeout_s: float,
    service_tier: Optional[str] = None,
) -> RunResult:
    messages = [
        {"role": "system", "content": add_cache_breaker(system_prompt)},
        {"role": "user", "content": user_message},
    ]
    params: Dict = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "stream": True,
    }
    if service_tier:
        params["service_tier"] = service_tier

    t0 = time.perf_counter()
    ttft_s = None
    ttfs_s = None
    content_parts = []
    
    try:
        stream = client.chat.completions.create(**params)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                if ttft_s is None:
                    ttft_s = time.perf_counter() - t0
                content_parts.append(chunk.choices[0].delta.content)
                
                # Check for first sentence completion
                if ttfs_s is None:
                    current_content = "".join(content_parts)
                    if detect_first_sentence_end(current_content):
                        ttfs_s = time.perf_counter() - t0
        
        total_latency_s = time.perf_counter() - t0
        content = "".join(content_parts)
        
        return RunResult(
            ttft_s=ttft_s or total_latency_s,
            ttfs_s=ttfs_s,
            total_latency_s=total_latency_s, 
            output_chars=len(content), 
            ok=True, 
            output=content
        )
    except Exception as e:
        total_latency_s = time.perf_counter() - t0
        return RunResult(
            ttft_s=total_latency_s,
            ttfs_s=None,
            total_latency_s=total_latency_s, 
            output_chars=0, 
            ok=False, 
            error=str(e)
        )


def run_openai_with_predicted(
    client: "OpenAI",
    model: str,
    system_prompt: str,
    user_message: str,
    predicted_text: str,
    timeout_s: float,
    service_tier: Optional[str] = None,
) -> RunResult:
    messages = [
        {"role": "system", "content": add_cache_breaker(system_prompt)},
        {"role": "user", "content": user_message},
    ]
    params: Dict = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "stream": True,
        "prediction": {"type": "content", "content": predicted_text},
    }
    if service_tier:
        params["service_tier"] = service_tier

    t0 = time.perf_counter()
    ttft_s = None
    ttfs_s = None
    content_parts = []
    
    try:
        stream = client.chat.completions.create(**params)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                if ttft_s is None:
                    ttft_s = time.perf_counter() - t0
                content_parts.append(chunk.choices[0].delta.content)
                
                # Check for first sentence completion
                if ttfs_s is None:
                    current_content = "".join(content_parts)
                    if detect_first_sentence_end(current_content):
                        ttfs_s = time.perf_counter() - t0
        
        total_latency_s = time.perf_counter() - t0
        content = "".join(content_parts)
        
        return RunResult(
            ttft_s=ttft_s or total_latency_s,
            ttfs_s=ttfs_s,
            total_latency_s=total_latency_s,
            output_chars=len(content),
            ok=True,
            output=content,
            draft_output=predicted_text
        )
    except Exception as e:
        total_latency_s = time.perf_counter() - t0
        return RunResult(
            ttft_s=total_latency_s,
            ttfs_s=None,
            total_latency_s=total_latency_s,
            output_chars=0,
            ok=False,
            error=str(e)
        )


def get_groq_draft_text(
    client: "Groq", model: str, system_prompt: str, user_message: str, max_tokens: Optional[int] = None
) -> Tuple[str, float, Optional[str]]:
    t0 = time.perf_counter()
    try:
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        resp = client.chat.completions.create(**params)
        delta = time.perf_counter() - t0
        text = resp.choices[0].message.content if resp and resp.choices else ""
        return text or "", delta, None
    except Exception as e:
        return "", time.perf_counter() - t0, str(e)


def generate_cache_breaker() -> str:
    """Generate a random word to inject into prompts to break caching."""
    random_word = ''.join(random.choices(string.ascii_lowercase, k=8))
    return f"[IGNORE: {random_word}] "

def add_cache_breaker(system_prompt: str) -> str:
    """Add cache breaker to start of system prompt."""
    return generate_cache_breaker() + system_prompt

def detect_first_sentence_end(text: str) -> bool:
    """Detect if the current text contains the end of the first sentence."""
    # Look for sentence-ending punctuation followed by space or end of string
    import re
    # Match . ! ? followed by whitespace or end of string, but not in abbreviations
    pattern = r'[.!?](?:\s|$)'
    matches = re.search(pattern, text)
    return matches is not None

def avg(values: List[float]) -> Optional[float]:
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def summarize_results(case_results: Dict[str, CaseAggregate]) -> Dict:
    summary = {}
    for fname, agg in case_results.items():
        def ok_ttft_times(results: List[RunResult]):
            return [r.ttft_s for r in results if r.ok]
        def ok_ttfs_times(results: List[RunResult]):
            return [r.ttfs_s for r in results if r.ok and r.ttfs_s is not None]
        def ok_total_latencies(results: List[RunResult]):
            return [r.total_latency_s for r in results if r.ok]
        def ok_draft_latencies(results: List[RunResult]):
            vals = [r.draft_latency_s for r in results if r.ok and r.draft_latency_s is not None]
            return [v for v in vals if isinstance(v, (int, float))]

        summary[fname] = {
            "groq_ttft_avg_s": avg(ok_ttft_times(agg.groq)),
            "openai_baseline_ttft_avg_s": avg(ok_ttft_times(agg.openai_baseline)),
            "openai_spec_ttft_avg_s": avg(ok_ttft_times(agg.openai_spec)),
            "groq_ttfs_avg_s": avg(ok_ttfs_times(agg.groq)),
            "openai_baseline_ttfs_avg_s": avg(ok_ttfs_times(agg.openai_baseline)),
            "openai_spec_ttfs_avg_s": avg(ok_ttfs_times(agg.openai_spec)),
            "groq_total_avg_s": avg(ok_total_latencies(agg.groq)),
            "openai_baseline_total_avg_s": avg(ok_total_latencies(agg.openai_baseline)),
            "openai_spec_total_avg_s": avg(ok_total_latencies(agg.openai_spec)),
            "openai_spec_draft_avg_s": avg(ok_draft_latencies(agg.openai_spec)),
            "runs": {
                "groq": [r.__dict__ for r in agg.groq],
                "openai_baseline": [r.__dict__ for r in agg.openai_baseline],
                "openai_spec": [r.__dict__ for r in agg.openai_spec],
            },
        }
    return summary


def emit_metrics_to_signoz(results: Dict[str, CaseAggregate], openai_model: str, groq_model: str):
    """Emit latency metrics to SignOz via OpenTelemetry"""
    if not report_metrics or not report_gauge or not inc_counter:
        print("OpenTelemetry not available, skipping metrics emission")
        return
    
    # Flatten results for easier processing
    all_results = []
    
    for fname, agg in results.items():
        # Process Groq results
        for run_idx, result in enumerate(agg.groq):
            attributes = {
                "deployment": "groq",
                "provider": "groq",
                "region": "groq-main",
                "success": str(result.ok).lower(),
                "model": f"direct-{groq_model}",
                "sdk_type": "direct_sdk",
                "test_case": fname,
                "run_index": str(run_idx)
            }
            
            if result.ok:
                # Report total latency
                report_metrics(
                    name="llm.deployment.latency",
                    instrument_type="histogram",
                    value=result.total_latency_s * 1000,  # Convert to ms
                    description="LLM deployment total latency in milliseconds",
                    attributes=attributes
                )
                
                # Report TTFT
                report_metrics(
                    name="llm.deployment.ttft",
                    instrument_type="histogram",
                    value=result.ttft_s * 1000,  # Convert to ms
                    description="LLM deployment time to first token in milliseconds",
                    attributes=attributes
                )
                
                # Report completion time if TTFS is available
                if result.ttfs_s is not None:
                    completion_time_ms = (result.total_latency_s - result.ttft_s) * 1000
                    report_metrics(
                        name="llm.deployment.completion_time",
                        instrument_type="histogram",
                        value=completion_time_ms,
                        description="LLM deployment completion time (after first token) in milliseconds",
                        attributes=attributes
                    )
                
                # Also report as gauges
                report_gauge(
                    name="llm.deployment.latency.current",
                    value=result.total_latency_s * 1000,
                    description="Current LLM deployment latency in milliseconds",
                    attributes=attributes
                )
                
                report_gauge(
                    name="llm.deployment.ttft.current",
                    value=result.ttft_s * 1000,
                    description="Current LLM deployment TTFT in milliseconds",
                    attributes=attributes
                )
                
                # Count successful calls
                inc_counter(
                    name="llm.deployment.calls.success",
                    value=1,
                    description="Number of successful LLM deployment calls",
                    attributes=attributes
                )
            else:
                # Count failed calls
                inc_counter(
                    name="llm.deployment.calls.failed",
                    value=1,
                    description="Number of failed LLM deployment calls",
                    attributes={**attributes, "error": result.error or "unknown"}
                )
                
                # Report failure as max latency
                report_gauge(
                    name="llm.deployment.latency.current",
                    value=999999,
                    description="Current LLM deployment latency in milliseconds",
                    attributes=attributes
                )
        
        # Process OpenAI baseline results
        for run_idx, result in enumerate(agg.openai_baseline):
            attributes = {
                "deployment": "openai-baseline",
                "provider": "openai",
                "region": "openai-main",
                "success": str(result.ok).lower(),
                "model": f"direct-{openai_model}",
                "sdk_type": "direct_sdk",
                "test_case": fname,
                "run_index": str(run_idx)
            }
            
            if result.ok:
                # Report total latency
                report_metrics(
                    name="llm.deployment.latency",
                    instrument_type="histogram",
                    value=result.total_latency_s * 1000,
                    description="LLM deployment total latency in milliseconds",
                    attributes=attributes
                )
                
                # Report TTFT
                if result.ttft_s is not None:
                    report_metrics(
                        name="llm.deployment.ttft",
                        instrument_type="histogram",
                        value=result.ttft_s * 1000,
                        description="LLM deployment time to first token in milliseconds",
                        attributes=attributes
                    )
                
                # Report completion time if available
                if result.ttfs_s is not None:
                    completion_time_ms = (result.total_latency_s - result.ttft_s) * 1000
                    report_metrics(
                        name="llm.deployment.completion_time",
                        instrument_type="histogram",
                        value=completion_time_ms,
                        description="LLM deployment completion time (after first token) in milliseconds",
                        attributes=attributes
                    )
                
                # Also report as gauges
                report_gauge(
                    name="llm.deployment.latency.current",
                    value=result.total_latency_s * 1000,
                    description="Current LLM deployment latency in milliseconds",
                    attributes=attributes
                )
                
                if result.ttft_s is not None:
                    report_gauge(
                        name="llm.deployment.ttft.current",
                        value=result.ttft_s * 1000,
                        description="Current LLM deployment TTFT in milliseconds",
                        attributes=attributes
                    )
                
                # Count successful calls
                inc_counter(
                    name="llm.deployment.calls.success",
                    value=1,
                    description="Number of successful LLM deployment calls",
                    attributes=attributes
                )
            else:
                # Count failed calls
                inc_counter(
                    name="llm.deployment.calls.failed",
                    value=1,
                    description="Number of failed LLM deployment calls",
                    attributes={**attributes, "error": result.error or "unknown"}
                )
                
                report_gauge(
                    name="llm.deployment.latency.current",
                    value=999999,
                    description="Current LLM deployment latency in milliseconds",
                    attributes=attributes
                )
        
        # Process OpenAI speculative results
        for run_idx, result in enumerate(agg.openai_spec):
            attributes = {
                "deployment": "openai-speculative",
                "provider": "openai",
                "region": "openai-main",
                "success": str(result.ok).lower(),
                "model": f"direct-{openai_model}-spec",
                "sdk_type": "direct_sdk_speculative",
                "test_case": fname,
                "run_index": str(run_idx),
                "draft_model": f"direct-{groq_model}"
            }
            
            if result.ok:
                # Report total latency
                report_metrics(
                    name="llm.deployment.latency",
                    instrument_type="histogram",
                    value=result.total_latency_s * 1000,
                    description="LLM deployment total latency in milliseconds",
                    attributes=attributes
                )
                
                # Report TTFT
                if result.ttft_s is not None:
                    report_metrics(
                        name="llm.deployment.ttft",
                        instrument_type="histogram",
                        value=result.ttft_s * 1000,
                        description="LLM deployment time to first token in milliseconds",
                        attributes=attributes
                    )
                
                # Report completion time if available
                if result.ttfs_s is not None:
                    completion_time_ms = (result.total_latency_s - result.ttft_s) * 1000
                    report_metrics(
                        name="llm.deployment.completion_time",
                        instrument_type="histogram",
                        value=completion_time_ms,
                        description="LLM deployment completion time (after first token) in milliseconds",
                        attributes=attributes
                    )
                
                # Report draft latency if available
                if result.draft_latency_s is not None:
                    report_metrics(
                        name="llm.deployment.draft_latency",
                        instrument_type="histogram",
                        value=result.draft_latency_s * 1000,
                        description="LLM deployment draft generation latency in milliseconds",
                        attributes=attributes
                    )
                
                # Also report as gauges
                report_gauge(
                    name="llm.deployment.latency.current",
                    value=result.total_latency_s * 1000,
                    description="Current LLM deployment latency in milliseconds",
                    attributes=attributes
                )
                
                if result.ttft_s is not None:
                    report_gauge(
                        name="llm.deployment.ttft.current",
                        value=result.ttft_s * 1000,
                        description="Current LLM deployment TTFT in milliseconds",
                        attributes=attributes
                    )
                
                # Count successful calls
                inc_counter(
                    name="llm.deployment.calls.success",
                    value=1,
                    description="Number of successful LLM deployment calls",
                    attributes=attributes
                )
            else:
                # Count failed calls
                inc_counter(
                    name="llm.deployment.calls.failed",
                    value=1,
                    description="Number of failed LLM deployment calls",
                    attributes={**attributes, "error": result.error or "unknown"}
                )
                
                report_gauge(
                    name="llm.deployment.latency.current",
                    value=999999,
                    description="Current LLM deployment latency in milliseconds",
                    attributes=attributes
                )
    
    # Report overall statistics
    all_groq = []
    all_openai_baseline = []
    all_openai_spec = []
    
    for agg in results.values():
        all_groq.extend([r for r in agg.groq if r.ok])
        all_openai_baseline.extend([r for r in agg.openai_baseline if r.ok])
        all_openai_spec.extend([r for r in agg.openai_spec if r.ok])
    
    # Report aggregated stats for each deployment type
    for deployment_type, results_list, model_suffix in [
        ("groq", all_groq, groq_model),
        ("openai-baseline", all_openai_baseline, openai_model),
        ("openai-speculative", all_openai_spec, f"{openai_model}-spec")
    ]:
        if results_list:
            latencies = [r.total_latency_s * 1000 for r in results_list]
            ttfts = [r.ttft_s * 1000 for r in results_list if r.ttft_s is not None]
            
            # Latency stats
            report_gauge(
                name="llm.deployment.latency.min",
                value=min(latencies),
                description="Minimum latency across runs",
                attributes={"measurement": deployment_type, "model": f"direct-{model_suffix}"}
            )
            
            report_gauge(
                name="llm.deployment.latency.max",
                value=max(latencies),
                description="Maximum latency across runs",
                attributes={"measurement": deployment_type, "model": f"direct-{model_suffix}"}
            )
            
            report_gauge(
                name="llm.deployment.latency.avg",
                value=sum(latencies) / len(latencies),
                description="Average latency across runs",
                attributes={"measurement": deployment_type, "model": f"direct-{model_suffix}"}
            )
            
            # TTFT stats
            if ttfts:
                report_gauge(
                    name="llm.deployment.ttft.min",
                    value=min(ttfts),
                    description="Minimum TTFT across runs",
                    attributes={"measurement": deployment_type, "model": f"direct-{model_suffix}"}
                )
                
                report_gauge(
                    name="llm.deployment.ttft.max",
                    value=max(ttfts),
                    description="Maximum TTFT across runs",
                    attributes={"measurement": deployment_type, "model": f"direct-{model_suffix}"}
                )
                
                report_gauge(
                    name="llm.deployment.ttft.avg",
                    value=sum(ttfts) / len(ttfts),
                    description="Average TTFT across runs",
                    attributes={"measurement": deployment_type, "model": f"direct-{model_suffix}"}
                )
    
    # Calculate overall success rate
    total_runs = sum(len(agg.groq) + len(agg.openai_baseline) + len(agg.openai_spec) for agg in results.values())
    successful_runs = len(all_groq) + len(all_openai_baseline) + len(all_openai_spec)
    
    if total_runs > 0:
        report_gauge(
            name="llm.deployment.success_rate",
            value=(successful_runs / total_runs) * 100,
            description="Success rate percentage",
            attributes={"measurement": "direct_global"}
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding vs baseline OpenAI and Groq")
    parser.add_argument("--evals_dir", default=str(Path(__file__).resolve().parents[1] / "evals_to_bench"))
    parser.add_argument("--runs", type=int, default=3, help="Runs per test case per mode")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--use_files", action="store_true", default=False, help="Parse test cases from evals_to_bench instead of embedded cases")

    # OpenAI
    parser.add_argument("--openai_model", default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--openai_base_url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--openai_api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--openai_secret_name", default=os.getenv("OPENAI_SECRET_NAME", "openai-api-key-scale"))
    parser.add_argument("--openai_service_tier", default=os.getenv("OPENAI_SERVICE_TIER", "auto"))

    # Always run predicted-outputs speculative path by default (Groq as drafter)
    parser.add_argument("--spec_enabled", action="store_true", default=True)

    # Groq
    parser.add_argument("--groq_model", default=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    parser.add_argument("--groq_api_key", default=os.getenv("GROQ_API_KEY"))

    args = parser.parse_args()

    evals_dir = Path(args.evals_dir)
    
    if args.use_files:
        # Only check if directory exists when we actually need to use files
        if not evals_dir.exists():
            raise FileNotFoundError(f"Evals directory not found: {evals_dir}")
        test_cases = read_test_cases(evals_dir)
    else:
        if EMBEDDED_CASES:
            test_cases = [
                TestCase(file=case["name"], system_prompt=case["system_prompt"], user_message=case["user_message"])  # type: ignore
                for case in EMBEDDED_CASES
            ]
        else:
            # Fall back to reading from directory if no embedded cases
            if not evals_dir.exists():
                raise FileNotFoundError(f"Evals directory not found: {evals_dir}")
            test_cases = read_test_cases(evals_dir)
    if not test_cases:
        raise RuntimeError("No test cases found")

    # Shuffle to avoid any systematic bias
    random.shuffle(test_cases)

    # Initialize clients
    openai_client = None
    # Prefer loading via access_secret to mirror repo behavior
    resolved_openai_api_key = None
    if access_secret is not None:
        for secret_name in [args.openai_secret_name, "openai-api-key"]:
            try:
                val = access_secret(secret_name)
                if val:
                    resolved_openai_api_key = val
                    break
            except Exception:
                continue
    # Fallbacks if secrets are not available
    if not resolved_openai_api_key:
        resolved_openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if OpenAI is not None and resolved_openai_api_key:
        openai_client = OpenAI(api_key=resolved_openai_api_key, base_url=args.openai_base_url)

    groq_client = None
    # Prefer secrets for Groq as well
    resolved_groq_api_key = None
    if access_secret is not None:
        try:
            resolved_groq_api_key = access_secret("groq-api-key")
        except Exception:
            resolved_groq_api_key = None
    if not resolved_groq_api_key:
        resolved_groq_api_key = args.groq_api_key or os.getenv("GROQ_API_KEY")
    if Groq is not None and resolved_groq_api_key:
        groq_client = Groq(api_key=resolved_groq_api_key)

    # Warm-ups with unique prompts to avoid prompt caching
    if openai_client:
        warmup_openai(openai_client, args.openai_model, count=2, use_spec=False)
        if args.spec_enabled:
            warmup_openai(openai_client, args.openai_model, count=2, use_spec=True)

    if groq_client:
        warmup_groq(groq_client, args.groq_model, count=2)

    results: Dict[str, CaseAggregate] = {}

    for tc in test_cases:
        agg = CaseAggregate(groq=[], openai_baseline=[], openai_spec=[])

        for _ in range(args.runs):
            # Groq baseline
            if groq_client:
                agg.groq.append(
                    run_groq(
                        groq_client,
                        model=args.groq_model,
                        system_prompt=tc.system_prompt,
                        user_message=tc.user_message,
                        timeout_s=args.timeout,
                    )
                )

            # OpenAI baseline (gpt-4o only)
            if openai_client:
                agg.openai_baseline.append(
                    run_openai(
                        openai_client,
                        model=args.openai_model,
                        system_prompt=tc.system_prompt,
                        user_message=tc.user_message,
                        timeout_s=args.timeout,
                        service_tier=args.openai_service_tier,
                    )
                )

            # OpenAI with predicted outputs using Groq draft
            if openai_client and groq_client and args.spec_enabled:
                # Start single timer for entire spec process (TTFT includes draft time)
                spec_start = time.perf_counter()
                
                draft_text, draft_latency, draft_err = get_groq_draft_text(
                    groq_client, args.groq_model, tc.system_prompt, tc.user_message
                )
                
                # Draft is complete, now start OpenAI streaming
                draft_complete_time = time.perf_counter() - spec_start
                
                spec_result = run_openai_with_predicted(
                    openai_client,
                    model=args.openai_model,
                    system_prompt=tc.system_prompt,
                    user_message=tc.user_message,
                    predicted_text=draft_text,
                    timeout_s=args.timeout,
                    service_tier=args.openai_service_tier,
                )
                
                # Calculate total TTFT: draft time + OpenAI TTFT
                total_ttft_s = draft_complete_time + spec_result.ttft_s
                # Calculate total TTFS: draft time + OpenAI TTFS (if available)
                total_ttfs_s = None
                if spec_result.ttfs_s is not None:
                    total_ttfs_s = draft_complete_time + spec_result.ttfs_s
                total_latency_s = time.perf_counter() - spec_start
                
                # Override with combined timings
                spec_result.ttft_s = total_ttft_s
                spec_result.ttfs_s = total_ttfs_s
                spec_result.total_latency_s = total_latency_s
                spec_result.draft_latency_s = draft_latency
                if draft_err and not spec_result.error:
                    spec_result.error = f"draft_err: {draft_err}"
                agg.openai_spec.append(spec_result)

        results[tc.file] = agg

    summary = summarize_results(results)
    
    # Initialize OpenTelemetry if available
    if init_otel is not None:
        init_otel()
    
    # Emit metrics to SignOz
    emit_metrics_to_signoz(results, args.openai_model, args.groq_model)

    print(json.dumps(
        {
            "config": {
                "openai_model": args.openai_model,
                "openai_base_url": args.openai_base_url,
                "openai_service_tier": args.openai_service_tier,
                "openai_secret_name": args.openai_secret_name,
                "spec_enabled": args.spec_enabled,
                "spec_mode": "predicted_outputs_groq",
                "groq_model": args.groq_model,
                "runs": args.runs,
            },
            "per_case": summary,
        },
        indent=2,
        sort_keys=True,
    ))

    # Pretty table summaries - 3 separate tables
    if tabulate is not None:
        def fmt(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else "-"

        # Collect data for all tables
        table_data = []
        overall_vals = {
            "openai_ttft": [], "spec_ttft": [], "groq_ttft": [],
            "openai_ttfs": [], "spec_ttfs": [], "groq_ttfs": [],  
            "openai_total": [], "spec_total": [], "groq_total": []
        }
        
        for fname, data in summary.items():
            o_ttft = data.get("openai_baseline_ttft_avg_s")
            s_ttft = data.get("openai_spec_ttft_avg_s") 
            g_ttft = data.get("groq_ttft_avg_s")
            o_ttfs = data.get("openai_baseline_ttfs_avg_s")
            s_ttfs = data.get("openai_spec_ttfs_avg_s")
            g_ttfs = data.get("groq_ttfs_avg_s")
            o_total = data.get("openai_baseline_total_avg_s")
            s_total = data.get("openai_spec_total_avg_s")
            g_total = data.get("groq_total_avg_s")

            # Collect for averages
            if isinstance(o_ttft, (int, float)): overall_vals["openai_ttft"].append(o_ttft)
            if isinstance(s_ttft, (int, float)): overall_vals["spec_ttft"].append(s_ttft)
            if isinstance(g_ttft, (int, float)): overall_vals["groq_ttft"].append(g_ttft)
            if isinstance(o_ttfs, (int, float)): overall_vals["openai_ttfs"].append(o_ttfs)
            if isinstance(s_ttfs, (int, float)): overall_vals["spec_ttfs"].append(s_ttfs)
            if isinstance(g_ttfs, (int, float)): overall_vals["groq_ttfs"].append(g_ttfs)
            if isinstance(o_total, (int, float)): overall_vals["openai_total"].append(o_total)
            if isinstance(s_total, (int, float)): overall_vals["spec_total"].append(s_total)
            if isinstance(g_total, (int, float)): overall_vals["groq_total"].append(g_total)

            table_data.append({
                'fname': fname,
                'o_ttft': o_ttft, 's_ttft': s_ttft, 'g_ttft': g_ttft,
                'o_ttfs': o_ttfs, 's_ttfs': s_ttfs, 'g_ttfs': g_ttfs,
                'o_total': o_total, 's_total': s_total, 'g_total': g_total
            })

        # Table 1: TTFT (Time To First Token)
        print("\n=== TIME TO FIRST TOKEN (TTFT) ===")
        ttft_headers = ["case", "groq_ttft_s", "openai_ttft_s", "spec_ttft_s", "spec_vs_openai", "spec_vs_groq"]
        ttft_rows = []
        
        for row in table_data:
            ttft_speedup_openai = (row['o_ttft'] / row['s_ttft']) if isinstance(row['o_ttft'], (int, float)) and isinstance(row['s_ttft'], (int, float)) and row['s_ttft'] > 0 else None
            ttft_speedup_groq = (row['g_ttft'] / row['s_ttft']) if isinstance(row['g_ttft'], (int, float)) and isinstance(row['s_ttft'], (int, float)) and row['s_ttft'] > 0 else None
            
            ttft_rows.append([
                row['fname'],
                fmt(row['g_ttft']),
                fmt(row['o_ttft']), 
                fmt(row['s_ttft']),
                fmt(ttft_speedup_openai),
                fmt(ttft_speedup_groq)
            ])
        
        # TTFT Overall row
        ttft_overall = [
            "Overall",
            fmt(avg(overall_vals["groq_ttft"])),
            fmt(avg(overall_vals["openai_ttft"])),
            fmt(avg(overall_vals["spec_ttft"])),
            fmt((avg(overall_vals["openai_ttft"]) / avg(overall_vals["spec_ttft"])) if avg(overall_vals["openai_ttft"]) and avg(overall_vals["spec_ttft"]) and avg(overall_vals["spec_ttft"]) > 0 else None),
            fmt((avg(overall_vals["groq_ttft"]) / avg(overall_vals["spec_ttft"])) if avg(overall_vals["groq_ttft"]) and avg(overall_vals["spec_ttft"]) and avg(overall_vals["spec_ttft"]) > 0 else None)
        ]
        ttft_rows.append(ttft_overall)
        print(tabulate(ttft_rows, headers=ttft_headers, tablefmt="github"))

        # Table 2: TTFS (Time To First Sentence)  
        print("\n=== TIME TO FIRST SENTENCE (TTFS) ===")
        ttfs_headers = ["case", "groq_ttfs_s", "openai_ttfs_s", "spec_ttfs_s", "spec_vs_openai", "spec_vs_groq"]
        ttfs_rows = []
        
        for row in table_data:
            ttfs_speedup_openai = (row['o_ttfs'] / row['s_ttfs']) if isinstance(row['o_ttfs'], (int, float)) and isinstance(row['s_ttfs'], (int, float)) and row['s_ttfs'] > 0 else None
            ttfs_speedup_groq = (row['g_ttfs'] / row['s_ttfs']) if isinstance(row['g_ttfs'], (int, float)) and isinstance(row['s_ttfs'], (int, float)) and row['s_ttfs'] > 0 else None
            
            ttfs_rows.append([
                row['fname'],
                fmt(row['g_ttfs']),
                fmt(row['o_ttfs']),
                fmt(row['s_ttfs']),
                fmt(ttfs_speedup_openai),
                fmt(ttfs_speedup_groq)
            ])
        
        # TTFS Overall row
        ttfs_overall = [
            "Overall", 
            fmt(avg(overall_vals["groq_ttfs"])),
            fmt(avg(overall_vals["openai_ttfs"])),
            fmt(avg(overall_vals["spec_ttfs"])),
            fmt((avg(overall_vals["openai_ttfs"]) / avg(overall_vals["spec_ttfs"])) if avg(overall_vals["openai_ttfs"]) and avg(overall_vals["spec_ttfs"]) and avg(overall_vals["spec_ttfs"]) > 0 else None),
            fmt((avg(overall_vals["groq_ttfs"]) / avg(overall_vals["spec_ttfs"])) if avg(overall_vals["groq_ttfs"]) and avg(overall_vals["spec_ttfs"]) and avg(overall_vals["spec_ttfs"]) > 0 else None)
        ]
        ttfs_rows.append(ttfs_overall)
        print(tabulate(ttfs_rows, headers=ttfs_headers, tablefmt="github"))

        # Table 3: TTLB (Time To Last Byte / Total Latency)
        print("\n=== TIME TO LAST BYTE (TTLB) ===") 
        ttlb_headers = ["case", "groq_total_s", "openai_total_s", "spec_total_s", "spec_vs_openai", "spec_vs_groq"]
        ttlb_rows = []
        
        for row in table_data:
            ttlb_speedup_openai = (row['o_total'] / row['s_total']) if isinstance(row['o_total'], (int, float)) and isinstance(row['s_total'], (int, float)) and row['s_total'] > 0 else None
            ttlb_speedup_groq = (row['g_total'] / row['s_total']) if isinstance(row['g_total'], (int, float)) and isinstance(row['s_total'], (int, float)) and row['s_total'] > 0 else None
            
            ttlb_rows.append([
                row['fname'],
                fmt(row['g_total']),
                fmt(row['o_total']),
                fmt(row['s_total']),
                fmt(ttlb_speedup_openai),
                fmt(ttlb_speedup_groq)
            ])
        
        # TTLB Overall row
        ttlb_overall = [
            "Overall",
            fmt(avg(overall_vals["groq_total"])),
            fmt(avg(overall_vals["openai_total"])),
            fmt(avg(overall_vals["spec_total"])),
            fmt((avg(overall_vals["openai_total"]) / avg(overall_vals["spec_total"])) if avg(overall_vals["openai_total"]) and avg(overall_vals["spec_total"]) and avg(overall_vals["spec_total"]) > 0 else None),
            fmt((avg(overall_vals["groq_total"]) / avg(overall_vals["spec_total"])) if avg(overall_vals["groq_total"]) and avg(overall_vals["spec_total"]) and avg(overall_vals["spec_total"]) > 0 else None)
        ]
        ttlb_rows.append(ttlb_overall)
        print(tabulate(ttlb_rows, headers=ttlb_headers, tablefmt="github"))


if __name__ == "__main__":
    main()
