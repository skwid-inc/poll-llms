"""
Test script to verify metrics are being emitted to SignOz correctly.
Run this locally to test the integration before deploying to Render.
"""

import asyncio
import os
from poll_llm_deployments import LLMLatencyPoller, LatencyResult
from datetime import datetime
from secret_manager import access_secret
from otel import init_otel, report_gauge, inc_counter

async def test_metrics_emission():
    """Test that metrics are properly emitted to SignOz"""
    
    print("Initializing OpenTelemetry...")
    init_otel()
    
    print("\nTesting basic metric emission...")
    # Test basic metrics
    report_gauge("test.llm.monitor.gauge", 42.0, "Test gauge metric")
    inc_counter("test.llm.monitor.counter", 1, "Test counter metric")
    print("✓ Basic metrics emitted")
    
    print("\nTesting LLM latency poller...")
    # Create a mock result to test metric emission
    test_results = [
        LatencyResult(
            deployment="test-openai",
            provider="openai",
            latency_ms=150.5,
            timestamp=datetime.now(),
            success=True,
            tokens_used=2
        ),
        LatencyResult(
            deployment="test-azure-westus",
            provider="azure",
            latency_ms=220.3,
            timestamp=datetime.now(),
            success=True,
            tokens_used=2
        ),
        LatencyResult(
            deployment="test-azure-failed",
            provider="azure",
            latency_ms=10000.0,
            timestamp=datetime.now(),
            success=False,
            error="Connection timeout"
        )
    ]
    
    # Create a poller instance
    poller = LLMLatencyPoller(
        openai_api_key="test-key",  # Won't be used for this test
        model_name="gpt-4o"
    )
    
    # Emit metrics
    poller.emit_metrics_to_signoz(test_results)
    print("✓ Test metrics emitted")
    
    print("\nMetrics emitted successfully!")
    print("Check your SignOz dashboard for the following metrics:")
    print("  - test.llm.monitor.gauge")
    print("  - test.llm.monitor.counter")
    print("  - llm.deployment.latency")
    print("  - llm.deployment.latency.current")
    print("  - llm.deployment.calls.success")
    print("  - llm.deployment.calls.failed")
    print("  - llm.deployment.latency.min/max/avg")
    print("  - llm.deployment.success_rate")

async def test_real_deployment():
    """Test with actual API calls to one deployment"""
    
    print("\n" + "="*60)
    print("Testing real deployment (this will make actual API calls)...")
    print("="*60)
    
    # Get API key
    openai_api_key = os.getenv("OPENAI_API_KEY") or access_secret("openai-api-key-scale")
    
    if not openai_api_key or openai_api_key == "test-key":
        print("⚠️  No valid OpenAI API key found. Skipping real deployment test.")
        return
    
    # Create poller
    poller = LLMLatencyPoller(
        openai_api_key=openai_api_key,
        model_name="gpt-4o",
        test_prompt="Hi",
        max_tokens=1,
        timeout=10.0
    )
    
    # Test just OpenAI to minimize cost
    from poll_llm_deployments import DeploymentConfig
    test_config = DeploymentConfig(name="openai-test", provider="openai")
    
    print(f"Testing {test_config.name}...")
    result = await poller._test_deployment(test_config)
    
    print(f"Result: {'✓ Success' if result.success else '✗ Failed'}")
    print(f"Latency: {result.latency_ms:.2f} ms")
    
    # Emit the metric
    poller.emit_metrics_to_signoz([result])
    print("✓ Real deployment metric emitted")

async def main():
    """Run all tests"""
    await test_metrics_emission()
    
    # Ask before running real test
    response = input("\nRun test with real API call? (y/N): ")
    if response.lower() == 'y':
        await test_real_deployment()
    
    print("\n✅ All tests completed!")
    print("\nNote: Metrics may take a few seconds to appear in SignOz.")

if __name__ == "__main__":
    asyncio.run(main())