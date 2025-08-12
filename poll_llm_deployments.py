import asyncio
import time
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json
import os

import aiohttp
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain.schema import HumanMessage
import pandas as pd
from dotenv import load_dotenv
from secret_manager import access_secret
from otel import report_gauge, report_metrics, inc_counter, init_otel

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional imports for direct OpenAI SDK
try:
    from openai import AsyncOpenAI, OpenAI
except Exception as e:
    AsyncOpenAI = None  # type: ignore
    OpenAI = None  # type: ignore
    logger.warning(f"OpenAI SDK not available: {e}")

########################
# Connections are established once per client and reused across requests so
# timings reflect request/streaming latency rather than handshake setup.
########################

@dataclass
class DeploymentConfig:
    """Configuration for a deployment endpoint"""
    name: str
    provider: str  # 'azure' or 'openai'
    endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    api_version: Optional[str] = None
    model: Optional[str] = None  # Model to use (e.g., 'gpt-4o', 'gpt-4o-mini')
    use_direct_sdk: bool = False  # Use direct SDK instead of LangChain
    sdk_type: str = "async"  # "async" or "sync" when use_direct_sdk is True
    
@dataclass
class LatencyResult:
    """Result from a latency test"""
    deployment: str
    provider: str
    latency_ms: float  # Total time to complete response
    timestamp: datetime
    success: bool
    error: Optional[str] = None
    time_to_first_token_ms: Optional[float] = None  # Time to first token (TTFT)
    completion_time_ms: Optional[float] = None  # Time from first to last token
    model: Optional[str] = None  # Model used (e.g., 'gpt-4o', 'gpt-4o-mini')

class LLMLatencyPoller:
    """Polls multiple LLM deployments to find the fastest one"""
    
    # Azure endpoints to test - loaded from environment variables
    azure_endpoints = {}  # Will store: {endpoint: {'api_key': key, 'model': model, 'region': region}}
    
    # Load Azure endpoints and API keys from environment variables
    @classmethod
    def _load_azure_endpoints(cls):
        """Load Azure endpoints and API keys from environment variables"""
        if cls.azure_endpoints:  # Already loaded
            return
            
        endpoint_configs = [
            ('VF', 'gpt-4o', os.getenv('AZURE_OPENAI_ENDPOINT_VF'), os.getenv('AZURE_OPENAI_API_KEY_VF') or access_secret("azure-openai-api-key")),
            ('WESTUS', 'gpt-4o', os.getenv('AZURE_OPENAI_ENDPOINT_WESTUS'), os.getenv('AZURE_OPENAI_API_KEY_WESTUS')),
            ('EASTUS2', 'gpt-4o', os.getenv('AZURE_OPENAI_ENDPOINT_EASTUS2'), os.getenv('AZURE_OPENAI_API_KEY_EASTUS2')),
            ('EASTUS2-mini', 'gpt-4o-mini', os.getenv('AZURE_OPENAI_ENDPOINT_EASTUS2_MINI'), os.getenv('AZURE_OPENAI_API_KEY_EASTUS2')),
            ('EASTUS', 'gpt-4o', os.getenv('AZURE_OPENAI_ENDPOINT_EASTUS'), os.getenv('AZURE_OPENAI_API_KEY_EASTUS')),
            # ('AUSTRALIAEAST', 'gpt-4o', os.getenv('AZURE_OPENAI_ENDPOINT_AUSTRALIAEAST'), os.getenv('AZURE_OPENAI_API_KEY_AUSTRALIAEAST')),
            # ('AUSTRALIAEAST-mini', 'gpt-4o-mini', os.getenv('AZURE_OPENAI_ENDPOINT_AUSTRALIAEAST_MINI'), os.getenv('AZURE_OPENAI_API_KEY_AUSTRALIAEAST')),
            # ('BRAZILSOUTH', os.getenv('AZURE_OPENAI_ENDPOINT_BRAZILSOUTH'), os.getenv('AZURE_OPENAI_API_KEY_BRAZILSOUTH')),
            ('CANADAEAST', 'gpt-4o', os.getenv('AZURE_OPENAI_ENDPOINT_CANADAEAST'), os.getenv('AZURE_OPENAI_API_KEY_CANADAEAST')),
            ('CANADAEAST-mini', 'gpt-4o-mini', os.getenv('AZURE_OPENAI_ENDPOINT_CANADAEAST_MINI'), os.getenv('AZURE_OPENAI_API_KEY_CANADAEAST')),
            ('SOUTHCENTRALUS', 'gpt-4o', os.getenv('AZURE_OPENAI_ENDPOINT_SOUTHCENTRALUS'), os.getenv('AZURE_OPENAI_API_KEY_SOUTHCENTRALUS')),
            ('SOUTHCENTRALUS-mini', 'gpt-4o-mini', os.getenv('AZURE_OPENAI_ENDPOINT_SOUTHCENTRALUS_MINI'), os.getenv('AZURE_OPENAI_API_KEY_SOUTHCENTRALUS')),
            ('NORTHCENTRALUS', 'gpt-4o', os.getenv('AZURE_OPENAI_ENDPOINT_NORTHCENTRALUS'), os.getenv('AZURE_OPENAI_API_KEY_NORTHCENTRALUS')),
            ('WESTUS3', 'gpt-4o', os.getenv('AZURE_OPENAI_ENDPOINT_WESTUS3'), os.getenv('AZURE_OPENAI_API_KEY_WESTUS3')),
            ('WESTUS3-mini', 'gpt-4o-mini', os.getenv('AZURE_OPENAI_ENDPOINT_WESTUS3_MINI'), os.getenv('AZURE_OPENAI_API_KEY_WESTUS3')),
        ]
        
        for region, model, endpoint, api_key in endpoint_configs:
            if endpoint and api_key:
                cls.azure_endpoints[endpoint] = {
                    'api_key': api_key,
                    'model': model,
                    'region': region
                }
            else:
                logger.warning(f"Missing Azure OpenAI configuration for {region}")
    
    def __init__(self, 
                 openai_api_key: str,
                 model_name: str = "gpt-4o",
                 azure_api_version: str = "2024-08-01-preview", # 2024-11-20
                 test_prompt: str = "Hi",
                 max_tokens: int = 1,
                 timeout: float = 10.0,
                 max_concurrent_tests: int = 1):
        """
        Initialize the poller with API keys and configuration
        
        Args:
            openai_api_key: OpenAI API key
            model_name: Model to test (default: gpt-4o)
            azure_api_version: Azure API version
            test_prompt: Simple prompt for testing (default: "Hi")
            max_tokens: Max tokens for response (default: 1 for minimal cost)
            timeout: Timeout for each request in seconds
            max_concurrent_tests: Maximum number of concurrent deployment tests (default: 3)
        """
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.azure_api_version = azure_api_version
        self.test_prompt = test_prompt
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.results_history: List[LatencyResult] = []
        self.semaphore = asyncio.Semaphore(max_concurrent_tests)
        
        # Load Azure endpoints on initialization
        self._load_azure_endpoints()
        
        # Initialize persistent clients (one OpenAI client and one Azure client per endpoint)
        self.openai_client: ChatOpenAI = self._create_openai_client()
        self.azure_clients: Dict[str, AzureChatOpenAI] = {
            endpoint: self._create_azure_client(endpoint) for endpoint in self.azure_endpoints.keys()
        }
        
        # Initialize direct SDK clients if available
        self.openai_sdk_client = None  # type: Optional[Any]  # Async client
        self.openai_sync_sdk_client = None  # type: Optional[Any]  # Sync client
        
        if AsyncOpenAI is not None:
            self.openai_sdk_client = AsyncOpenAI(
                api_key=self.openai_api_key,
                timeout=self.timeout,
                max_retries=0  # No retries for accurate timing
            )
        
        if OpenAI is not None:
            self.openai_sync_sdk_client = OpenAI(
                api_key=self.openai_api_key,
                timeout=self.timeout,
                max_retries=0  # No retries for accurate timing
            )
        
    def _create_openai_client(self, model: Optional[str] = None) -> ChatOpenAI:
        """Create OpenAI client"""
        return ChatOpenAI(
            api_key=self.openai_api_key,
            model=model or self.model_name,
            temperature=0,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=0,  # No retries for accurate timing
            streaming=True  # Always use streaming
        )
        
    def _create_azure_client(self, endpoint: str, model: Optional[str] = None) -> AzureChatOpenAI:
        """Create Azure OpenAI client for a specific endpoint"""
        # Use the model from config if provided, otherwise use the model from endpoint config
        endpoint_config = self.azure_endpoints.get(endpoint, {})
        deployment_model = model or endpoint_config.get('model', self.model_name)
        api_key = endpoint_config.get('api_key') if isinstance(endpoint_config, dict) else endpoint_config
        
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=deployment_model,
            api_version=self.azure_api_version,
            api_key=api_key,
            temperature=0,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=0,
            streaming=True  # Always use streaming
        )
        
    async def _test_deployment(self, 
                              deployment_config: DeploymentConfig) -> LatencyResult:
        """
        Test a single deployment and measure latency
        
        Args:
            deployment_config: Configuration for the deployment to test
            
        Returns:
            LatencyResult with timing information including TTFT
        """
        async with self.semaphore:  # Limit concurrent tests
            timestamp = datetime.now()
            start_time = None
            first_token_time = None
            full_response = ""
            
            try:
                if deployment_config.provider == 'openai' and deployment_config.use_direct_sdk and deployment_config.sdk_type == "sync" and self.openai_sync_sdk_client:
                    # Use synchronous OpenAI SDK (copying pattern from poll_direct_deployments.py)
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": self.test_prompt}
                    ]
                    
                    # Begin timing just before sending the request
                    start_time = time.perf_counter()
                    
                    # Create stream synchronously
                    stream = self.openai_sync_sdk_client.chat.completions.create(
                        model=deployment_config.model or self.model_name,
                        messages=messages,
                        temperature=0,
                        max_tokens=self.max_tokens,
                        stream=True
                    )
                    
                    # Iterate synchronously (pattern from poll_direct_deployments.py)
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content is not None:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            full_response += chunk.choices[0].delta.content
                
                elif deployment_config.provider == 'openai' and deployment_config.use_direct_sdk and self.openai_sdk_client:
                    # Use direct OpenAI SDK
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": self.test_prompt}
                    ]
                    
                    # Begin timing just before sending the request
                    start_time = time.perf_counter()
                    
                    # Stream the response using direct SDK
                    stream = await self.openai_sdk_client.chat.completions.create(
                        model=deployment_config.model or self.model_name,
                        messages=messages,
                        temperature=0,
                        max_tokens=self.max_tokens,
                        stream=True
                    )
                    
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content is not None:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            full_response += chunk.choices[0].delta.content
                
                else:
                    # Use LangChain clients
                    if deployment_config.provider == 'openai':
                        # Create a new client for OpenAI with the specific model
                        client = self._create_openai_client(model=deployment_config.model)
                    else:  # azure
                        # Safe-guard in case an endpoint appears without a prepared client
                        if deployment_config.endpoint in self.azure_clients:
                            client = self.azure_clients[deployment_config.endpoint]
                        else:
                            client = self._create_azure_client(deployment_config.endpoint, model=deployment_config.model)
                    
                    # Create a simple message
                    message = HumanMessage(content=self.test_prompt)
                    
                    # Begin timing just before sending the request
                    start_time = time.perf_counter()
                    
                    # Stream the response to measure TTFT
                    async for chunk in client.astream([message]):
                        has_payload = (getattr(chunk, "content", None) not in (None, "")) or (
                            hasattr(chunk, "tool_calls") and chunk.tool_calls
                        )
                        if first_token_time is None and has_payload:
                            first_token_time = time.perf_counter()

                        if getattr(chunk, "content", None):
                            full_response += chunk.content
                
                # Calculate all timing metrics
                end_time = time.perf_counter()
                total_latency_ms = (end_time - start_time) * 1000
                
                if first_token_time:
                    ttft_ms = (first_token_time - start_time) * 1000
                    completion_time_ms = (end_time - first_token_time) * 1000
                else:
                    # No tokens received
                    ttft_ms = total_latency_ms
                    completion_time_ms = 0
                
                return LatencyResult(
                    deployment=deployment_config.name,
                    provider=deployment_config.provider,
                    latency_ms=total_latency_ms,
                    timestamp=timestamp,
                    success=True,
                    time_to_first_token_ms=ttft_ms,
                    completion_time_ms=completion_time_ms,
                    model=deployment_config.model
                )
                
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000 if start_time else 0
                logger.warning(f"Error testing {deployment_config.name}: {str(e)}")
                
                return LatencyResult(
                    deployment=deployment_config.name,
                    provider=deployment_config.provider,
                    latency_ms=latency_ms,
                    timestamp=timestamp,
                    success=False,
                    error=str(e),
                    time_to_first_token_ms=None,
                    completion_time_ms=None,
                    model=deployment_config.model
                )
    
    async def _warm_client(self, client: ChatOpenAI) -> None:
        """Send a minimal request to warm up the underlying HTTP connection."""
        try:
            message = HumanMessage(content="ping")
            # Stream just until first token to establish connection; discard timing
            async for _ in client.astream([message]):
                break
        except Exception as e:
            logger.warning(f"Warm-up failed: {str(e)}")
            # Continue without failing the overall run
    
    async def _warm_sdk_client(self, model: str) -> None:
        """Send a minimal request to warm up the direct SDK HTTP connection."""
        if not self.openai_sdk_client:
            return
            
        try:
            messages = [{"role": "user", "content": "ping"}]
            stream = await self.openai_sdk_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=1,
                stream=True
            )
            # Read just first chunk to establish connection
            async for _ in stream:
                break
        except Exception as e:
            logger.warning(f"SDK warm-up failed: {str(e)}")
            # Continue without failing the overall run
    
    async def _warm_sync_sdk_client(self, model: str) -> None:
        """Send a minimal request to warm up the sync SDK HTTP connection."""
        if not self.openai_sync_sdk_client:
            return
            
        try:
            messages = [{"role": "user", "content": "ping"}]
            # Run sync warm-up in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._warm_sync_sdk_client_sync,
                model,
                messages
            )
        except Exception as e:
            logger.warning(f"Sync SDK warm-up failed: {str(e)}")
            # Continue without failing the overall run
    
    def _warm_sync_sdk_client_sync(self, model: str, messages: list) -> None:
        """Synchronous helper for warming up sync SDK client."""
        try:
            stream = self.openai_sync_sdk_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=1,
                stream=True
            )
            # Read just first chunk to establish connection
            for _ in stream:
                break
        except Exception:
            pass
    
    async def poll_all_deployments(self) -> List[LatencyResult]:
        """
        Poll all configured deployments concurrently
        
        Returns:
            List of LatencyResult objects
        """
        deployments = []
        
        # Add OpenAI deployments - one for gpt-4o and one for gpt-4o-mini
        # LangChain versions
        deployments.append(
            DeploymentConfig(
                name="openai-langchain-gpt-4o",
                provider="openai",
                model="gpt-4o",
                use_direct_sdk=False
            )
        )
        deployments.append(
            DeploymentConfig(
                name="openai-langchain-gpt-4o-mini",
                provider="openai",
                model="gpt-4o-mini",
                use_direct_sdk=False
            )
        )
        
        # Direct async SDK versions (if available)
        if self.openai_sdk_client:
            deployments.append(
                DeploymentConfig(
                    name="openai-async-sdk-gpt-4o",
                    provider="openai",
                    model="gpt-4o",
                    use_direct_sdk=True,
                    sdk_type="async"
                )
            )
            deployments.append(
                DeploymentConfig(
                    name="openai-async-sdk-gpt-4o-mini",
                    provider="openai",
                    model="gpt-4o-mini",
                    use_direct_sdk=True,
                    sdk_type="async"
                )
            )
        
        # Direct sync SDK versions (if available)
        if self.openai_sync_sdk_client:
            deployments.append(
                DeploymentConfig(
                    name="openai-sync-sdk-gpt-4o",
                    provider="openai",
                    model="gpt-4o",
                    use_direct_sdk=True,
                    sdk_type="sync"
                )
            )
            deployments.append(
                DeploymentConfig(
                    name="openai-sync-sdk-gpt-4o-mini",
                    provider="openai",
                    model="gpt-4o-mini",
                    use_direct_sdk=True,
                    sdk_type="sync"
                )
            )
        
        # Add Azure deployments for each region with their configured models
        for endpoint, config in self.azure_endpoints.items():
            region = config.get('region', 'unknown')
            model = config.get('model', self.model_name)
            deployments.append(
                DeploymentConfig(
                    name=f"azure-{region}-{model}",
                    provider="azure",
                    endpoint=endpoint,
                    model=model
                )
            )
        
        # Warm connections before timing so handshake is not included
        # Create temporary OpenAI clients for warming up both models
        openai_4o_client = self._create_openai_client(model="gpt-4o")
        openai_4o_mini_client = self._create_openai_client(model="gpt-4o-mini")
        
        warm_tasks = [
            self._warm_client(openai_4o_client),
            self._warm_client(openai_4o_mini_client),
            *[self._warm_client(client) for client in self.azure_clients.values()],
        ]
        
        # Also warm up direct SDK connections if available
        if self.openai_sdk_client:
            warm_tasks.extend([
                self._warm_sdk_client("gpt-4o"),
                self._warm_sdk_client("gpt-4o-mini")
            ])
        
        # Also warm up sync SDK connections if available
        if self.openai_sync_sdk_client:
            warm_tasks.extend([
                self._warm_sync_sdk_client("gpt-4o"),
                self._warm_sync_sdk_client("gpt-4o-mini")
            ])
        
        await asyncio.gather(*warm_tasks)
        
        # Test all deployments concurrently with semaphore limiting concurrency
        # This prevents bandwidth-induced queueing delays that would affect all tasks equally
        logger.info(f"Testing {len(deployments)} deployments with max concurrency of {self.semaphore._value}...")
        tasks = [self._test_deployment(config) for config in deployments]
        results = await asyncio.gather(*tasks)
        
        # Store results in history
        self.results_history.extend(results)
        
        return results
    
    def get_fastest_deployment(self, 
                              results: Optional[List[LatencyResult]] = None,
                              provider_filter: Optional[str] = None) -> Optional[LatencyResult]:
        """
        Get the fastest deployment from results
        
        Args:
            results: List of results to analyze (uses latest if None)
            provider_filter: Filter by provider ('azure' or 'openai')
            
        Returns:
            LatencyResult of the fastest deployment
        """
        if results is None:
            # Get the most recent results for each deployment
            latest_results = {}
            for result in reversed(self.results_history):
                if result.deployment not in latest_results:
                    latest_results[result.deployment] = result
            results = list(latest_results.values())
        
        # Filter successful results
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return None
            
        # Apply provider filter if specified
        if provider_filter:
            successful_results = [r for r in successful_results 
                                 if r.provider == provider_filter]
        
        if not successful_results:
            return None
            
        # Find the fastest
        return min(successful_results, key=lambda r: r.latency_ms)
    
    def print_results_summary(self, results: List[LatencyResult]):
        """Print a formatted summary of results"""
        print("\n" + "="*100)
        print(f"Latency Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)
        
        # Separate by provider
        openai_results = [r for r in results if r.provider == 'openai']
        azure_results = [r for r in results if r.provider == 'azure']
        
        # Print OpenAI results
        if openai_results:
            # Separate SDK and LangChain results
            langchain_results = [r for r in openai_results if 'langchain' in r.deployment]
            async_sdk_results = [r for r in openai_results if 'async-sdk' in r.deployment]
            sync_sdk_results = [r for r in openai_results if 'sync-sdk' in r.deployment]
            
            if langchain_results:
                print("\nOpenAI Deployments (LangChain):")
                print("-"*100)
                print(f"{'Deployment':40s} {'Total':>10s} {'TTFT':>10s} {'Completion':>12s} {'Status':>8s}")
                print("-"*100)
                for r in langchain_results:
                    status = "✓" if r.success else "✗"
                    if r.success and r.time_to_first_token_ms:
                        print(f"{r.deployment:40s} {r.latency_ms:10.2f} {r.time_to_first_token_ms:10.2f} {r.completion_time_ms:12.2f} {status:>8s}")
                    else:
                        print(f"{r.deployment:40s} {r.latency_ms:10.2f} {'N/A':>10s} {'N/A':>12s} {status:>8s}")
            
            if async_sdk_results:
                print("\nOpenAI Deployments (Async SDK):")
                print("-"*100)
                print(f"{'Deployment':40s} {'Total':>10s} {'TTFT':>10s} {'Completion':>12s} {'Status':>8s}")
                print("-"*100)
                for r in async_sdk_results:
                    status = "✓" if r.success else "✗"
                    if r.success and r.time_to_first_token_ms:
                        print(f"{r.deployment:40s} {r.latency_ms:10.2f} {r.time_to_first_token_ms:10.2f} {r.completion_time_ms:12.2f} {status:>8s}")
                    else:
                        print(f"{r.deployment:40s} {r.latency_ms:10.2f} {'N/A':>10s} {'N/A':>12s} {status:>8s}")
            
            if sync_sdk_results:
                print("\nOpenAI Deployments (Sync SDK):")
                print("-"*100)
                print(f"{'Deployment':40s} {'Total':>10s} {'TTFT':>10s} {'Completion':>12s} {'Status':>8s}")
                print("-"*100)
                for r in sync_sdk_results:
                    status = "✓" if r.success else "✗"
                    if r.success and r.time_to_first_token_ms:
                        print(f"{r.deployment:40s} {r.latency_ms:10.2f} {r.time_to_first_token_ms:10.2f} {r.completion_time_ms:12.2f} {status:>8s}")
                    else:
                        print(f"{r.deployment:40s} {r.latency_ms:10.2f} {'N/A':>10s} {'N/A':>12s} {status:>8s}")
        
        # Print Azure results (sorted by TTFT)
        if azure_results:
            print("\nAzure Deployments:")
            print("-"*100)
            print(f"{'Deployment':30s} {'Total':>10s} {'TTFT':>10s} {'Completion':>12s} {'Status':>8s}")
            print("-"*100)
            azure_sorted = sorted(azure_results, key=lambda r: r.time_to_first_token_ms if r.success and r.time_to_first_token_ms else float('inf'))
            for r in azure_sorted:
                status = "✓" if r.success else "✗"
                if r.success and r.time_to_first_token_ms:
                    print(f"{r.deployment:30s} {r.latency_ms:10.2f} {r.time_to_first_token_ms:10.2f} {r.completion_time_ms:12.2f} {status:>8s}")
                else:
                    error_msg = f" ({r.error[:20]}...)" if r.error and len(r.error) > 20 else f" ({r.error})" if r.error else ""
                    print(f"{r.deployment:30s} {r.latency_ms:10.2f} {'N/A':>10s} {'N/A':>12s} {status:>8s}{error_msg}")
        
        # Print fastest by different metrics
        successful = [r for r in results if r.success and r.time_to_first_token_ms]
        if successful:
            print("\n" + "="*60)
            print("🏆 Performance Champions:")
            print("-"*60)
            
            # Fastest TTFT
            fastest_ttft = min(successful, key=lambda r: r.time_to_first_token_ms)
            print(f"Fastest Time to First Token: {fastest_ttft.deployment}")
            print(f"   TTFT: {fastest_ttft.time_to_first_token_ms:.2f} ms")
            
            # Fastest total completion
            fastest_total = min(successful, key=lambda r: r.latency_ms)
            print(f"\nFastest Total Completion: {fastest_total.deployment}")
            print(f"   Total: {fastest_total.latency_ms:.2f} ms")
        
        # Calculate statistics
        if successful:
            latencies = [r.latency_ms for r in successful]
            ttfts = [r.time_to_first_token_ms for r in successful]
            
            print("\n" + "="*60)
            print("Statistics:")
            print(f"  Success Rate: {len(successful)}/{len(results)} ({100*len(successful)/len(results):.1f}%)")
            print(f"\n  Total Latency (ms):")
            print(f"    Min: {min(latencies):.2f}, Max: {max(latencies):.2f}, Avg: {sum(latencies)/len(latencies):.2f}")
            print(f"\n  Time to First Token (ms):")
            print(f"    Min: {min(ttfts):.2f}, Max: {max(ttfts):.2f}, Avg: {sum(ttfts)/len(ttfts):.2f}")
            
            # Compare SDK vs LangChain if both are available
            sdk_openai = [r for r in successful if r.provider == 'openai' and 'sdk' in r.deployment]
            langchain_openai = [r for r in successful if r.provider == 'openai' and 'langchain' in r.deployment]
            
            if sdk_openai and langchain_openai:
                print("\n" + "="*60)
                print("🔍 SDK vs LangChain Comparison (OpenAI):")
                print("-"*60)
                
                # Group by model
                models = set(r.model for r in sdk_openai + langchain_openai if r.model)
                for model in sorted(models):
                    sdk_model = [r for r in sdk_openai if r.model == model]
                    lc_model = [r for r in langchain_openai if r.model == model]
                    
                    if sdk_model and lc_model:
                        sdk_ttft = sdk_model[0].time_to_first_token_ms
                        lc_ttft = lc_model[0].time_to_first_token_ms
                        sdk_total = sdk_model[0].latency_ms
                        lc_total = lc_model[0].latency_ms
                        
                        print(f"\n  Model: {model}")
                        print(f"    TTFT: SDK={sdk_ttft:.2f}ms, LangChain={lc_ttft:.2f}ms (SDK is {abs(sdk_ttft-lc_ttft):.2f}ms {'faster' if sdk_ttft < lc_ttft else 'slower'})")
                        print(f"    Total: SDK={sdk_total:.2f}ms, LangChain={lc_total:.2f}ms (SDK is {abs(sdk_total-lc_total):.2f}ms {'faster' if sdk_total < lc_total else 'slower'})")
    
    def emit_metrics_to_signoz(self, results: List[LatencyResult]):
        """Emit latency metrics to SignOz via OpenTelemetry"""
        for result in results:
            # Extract region from deployment name
            region = "unknown"
            if result.provider == "azure":
                # Parse region from deployment name like "azure-WESTUS-gpt-4o"
                parts = result.deployment.split('-')
                if len(parts) >= 2:
                    region = parts[1].lower()
            elif result.provider == "openai":
                region = "openai-main"
            
            # Detect SDK type
            sdk_type = "unknown"
            if "async-sdk" in result.deployment:
                sdk_type = "async_sdk"
            elif "sync-sdk" in result.deployment:
                sdk_type = "sync_sdk"
            elif "langchain" in result.deployment:
                sdk_type = "langchain"
            elif result.provider == "azure":
                sdk_type = "langchain"  # Azure always uses LangChain
            
            attributes = {
                "deployment": result.deployment,
                "provider": result.provider,
                "region": region,
                "success": str(result.success).lower(),
                "model": result.model or "unknown",
                "sdk_type": sdk_type
            }
            
            if result.success:
                # Report total latency as a histogram (for percentiles)
                report_metrics(
                    name="llm.deployment.latency",
                    instrument_type="histogram",
                    value=result.latency_ms,
                    description="LLM deployment total latency in milliseconds",
                    attributes=attributes
                )
                
                # Report TTFT if available
                if result.time_to_first_token_ms is not None:
                    report_metrics(
                        name="llm.deployment.ttft",
                        instrument_type="histogram",
                        value=result.time_to_first_token_ms,
                        description="LLM deployment time to first token in milliseconds",
                        attributes=attributes
                    )
                    
                    # Report completion time
                    report_metrics(
                        name="llm.deployment.completion_time",
                        instrument_type="histogram",
                        value=result.completion_time_ms,
                        description="LLM deployment completion time (after first token) in milliseconds",
                        attributes=attributes
                    )
                
                # Also report as gauges for current values
                report_gauge(
                    name="llm.deployment.latency.current",
                    value=result.latency_ms,
                    description="Current LLM deployment latency in milliseconds",
                    attributes=attributes
                )
                
                if result.time_to_first_token_ms is not None:
                    report_gauge(
                        name="llm.deployment.ttft.current",
                        value=result.time_to_first_token_ms,
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
                
                # Report failure as max latency for monitoring
                report_gauge(
                    name="llm.deployment.latency.current",
                    value=999999,  # High value to indicate failure
                    description="Current LLM deployment latency in milliseconds",
                    attributes=attributes
                )
        
        # Report overall statistics
        successful_results = [r for r in results if r.success]
        if successful_results:
            latencies = [r.latency_ms for r in successful_results]
            ttfts = [r.time_to_first_token_ms for r in successful_results if r.time_to_first_token_ms is not None]
            
            # Total latency stats
            report_gauge(
                name="llm.deployment.latency.min",
                value=min(latencies),
                description="Minimum latency across all deployments",
                attributes={"measurement": "global"}
            )
            
            report_gauge(
                name="llm.deployment.latency.max",
                value=max(latencies),
                description="Maximum latency across all deployments",
                attributes={"measurement": "global"}
            )
            
            report_gauge(
                name="llm.deployment.latency.avg",
                value=sum(latencies) / len(latencies),
                description="Average latency across all deployments",
                attributes={"measurement": "global"}
            )
            
            # TTFT stats
            if ttfts:
                report_gauge(
                    name="llm.deployment.ttft.min",
                    value=min(ttfts),
                    description="Minimum TTFT across all deployments",
                    attributes={"measurement": "global"}
                )
                
                report_gauge(
                    name="llm.deployment.ttft.max",
                    value=max(ttfts),
                    description="Maximum TTFT across all deployments",
                    attributes={"measurement": "global"}
                )
                
                report_gauge(
                    name="llm.deployment.ttft.avg",
                    value=sum(ttfts) / len(ttfts),
                    description="Average TTFT across all deployments",
                    attributes={"measurement": "global"}
                )
            
            report_gauge(
                name="llm.deployment.success_rate",
                value=(len(successful_results) / len(results)) * 100,
                description="Success rate percentage",
                attributes={"measurement": "global"}
            )
    
    def save_results_to_csv(self, filename: str = "latency_results.csv"):
        """Save all historical results to a CSV file"""
        if not self.results_history:
            logger.warning("No results to save")
            return
            
        # Convert to DataFrame
        data = []
        for r in self.results_history:
            data.append({
                'timestamp': r.timestamp,
                'deployment': r.deployment,
                'provider': r.provider,
                'model': r.model or '',
                'latency_ms': r.latency_ms,
                'time_to_first_token_ms': r.time_to_first_token_ms,
                'completion_time_ms': r.completion_time_ms,
                'success': r.success,
                'error': r.error or ''
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")
    
    async def continuous_polling(self, 
                                interval_seconds: int = 60,
                                duration_minutes: Optional[int] = None):
        """
        Continuously poll deployments at regular intervals
        
        Args:
            interval_seconds: Seconds between polls
            duration_minutes: Total duration to run (None for infinite)
        """
        start_time = time.time()
        poll_count = 0
        
        while True:
            poll_count += 1
            logger.info(f"Starting poll #{poll_count}")
            
            # Poll all deployments
            results = await self.poll_all_deployments()
            
            # Emit metrics to SignOz
            self.emit_metrics_to_signoz(results)
            
            # Print summary
            self.print_results_summary(results)
            
            # Save to CSV periodically (every 10 polls)
            # if poll_count % 10 == 0:
                # self.save_results_to_csv(f"latency_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            
            # Check if we should stop
            if duration_minutes:
                elapsed_minutes = (time.time() - start_time) / 60
                if elapsed_minutes >= duration_minutes:
                    logger.info(f"Reached duration limit of {duration_minutes} minutes")
                    break
            
            # Wait for next interval
            logger.info(f"Waiting {interval_seconds} seconds until next poll...")
            await asyncio.sleep(interval_seconds)

    def display_results_table(self, all_runs: List[List[LatencyResult]]):
        """Display results from multiple runs in a table format"""
        # Prepare data for DataFrame
        data = []
        deployments = set()
        
        for run_idx, results in enumerate(all_runs):
            for r in results:
                deployments.add(r.deployment)
                data.append({
                    'Run': run_idx + 1,
                    'Deployment': r.deployment,
                    'Provider': r.provider,
                    'Latency (ms)': round(r.latency_ms, 2) if r.success else 'Failed',
                    'Success': '✓' if r.success else '✗',
                    'Error': r.error[:50] + '...' if r.error and len(r.error) > 50 else (r.error or '')
                })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Create pivot table for better visualization
        pivot_df = df.pivot_table(
            index='Deployment',
            columns='Run',
            values='Latency (ms)',
            aggfunc='first'
        )
        
        print("\n" + "="*80)
        print("LATENCY RESULTS ACROSS RUNS (ms)")
        print("="*80)
        print(pivot_df.to_string())
        
        # Calculate statistics
        print("\n" + "-"*80)
        print("STATISTICS PER DEPLOYMENT")
        print("-"*80)
        
        for deployment in sorted(deployments):
            dep_data = [r for run in all_runs for r in run if r.deployment == deployment]
            successful = [r for r in dep_data if r.success]
            
            if successful:
                latencies = [r.latency_ms for r in successful]
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                success_rate = len(successful) / len(dep_data) * 100
                
                print(f"\n{deployment}:")
                print(f"  Success Rate: {success_rate:.1f}% ({len(successful)}/{len(dep_data)})")
                print(f"  Avg Latency: {avg_latency:.2f} ms")
                print(f"  Min Latency: {min_latency:.2f} ms")
                print(f"  Max Latency: {max_latency:.2f} ms")
            else:
                print(f"\n{deployment}:")
                print(f"  All attempts failed")
                if dep_data and dep_data[0].error:
                    print(f"  Error: {dep_data[0].error}")


async def main():
    """Main function to run the latency poller"""
    
    # Initialize OpenTelemetry
    init_otel()
    
    # Get API keys from environment or your secret manager
    openai_api_key = access_secret("openai-api-key-scale")
    
    # Get max concurrent tests from environment or use default
    max_concurrent_tests = int(os.getenv("MAX_CONCURRENT_TESTS", "1"))
    logger.info(f"Using max concurrent tests: {max_concurrent_tests}")
    
    # Create poller instance
    # Using max_concurrent_tests to limit bandwidth usage and get more accurate absolute timing
    poller = LLMLatencyPoller(
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini",
        test_prompt="Hi",  # Minimal prompt to reduce cost
        max_tokens=1,  # Single token response
        timeout=10.0,
        max_concurrent_tests=max_concurrent_tests  # Limit concurrency to avoid bandwidth-induced queueing delays
    )
    
    # Check if we're running as a cron job or continuous mode
    run_mode = os.getenv("RUN_MODE", "single")  # "single" for cron, "continuous" for testing
    
    # if run_mode == "continuous":
        # Run continuously (for testing)
    await poller.continuous_polling(
        interval_seconds=5,  # 10 minutes
        duration_minutes=None  # Run indefinitely
    )
    # else:
    #     # Run once and exit (for cron job)
    #     logger.info("Running single latency test across all deployments...")
        
    #     # Poll all deployments
    #     results = await poller.poll_all_deployments()
        
    #     # Emit metrics to SignOz
    #     poller.emit_metrics_to_signoz(results)
        
    #     # Print summary for logs
    #     poller.print_results_summary(results)
        
    #     # Log completion
    #     logger.info("Latency test completed and metrics emitted to SignOz")


if __name__ == "__main__":
    asyncio.run(main())