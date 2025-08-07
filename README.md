# Poll LLMs

This project monitors and benchmarks latency across multiple LLM deployment endpoints.

## Setup

### Environment Variables

1. Copy the example environment file:
   ```bash
   cp env.example .env
   ```

2. Edit `.env` and replace the placeholder values with your actual API keys:
   - For the VF endpoint, replace `your_azure_vf_api_key_here` with your actual API key or keep it to use `access_secret("azure-openai-api-key")`
   - Other API keys are already provided in the example file

3. The environment variables are loaded automatically when running the scripts.

### Running the Scripts

- **Simple latency test**: `python poll_llm_deployments.py`
- **Comprehensive test with tools**: `python long_poll_llm_deployments.py`

Both scripts support different run modes via environment variables:
- `RUN_MODE=single` (default) - Run once and exit (for cron jobs)
- `RUN_MODE=continuous` - Run continuously at intervals
- `USE_PARALLEL=true` (default for long_poll) - Run parallel requests per deployment
- `NUM_REQUESTS_PER_DEPLOYMENT=5` - Number of parallel requests per deployment
- `MAX_CONCURRENT_TESTS=3` (default) - Maximum number of deployments to test simultaneously. This prevents bandwidth-induced queueing delays when testing multiple endpoints

## Monitoring

See [README_MONITORING.md](README_MONITORING.md) for detailed monitoring setup instructions.
