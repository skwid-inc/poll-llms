# LLM Deployment Latency Monitoring

This service monitors the latency of various LLM deployments (OpenAI and Azure OpenAI) and emits metrics to SignOz for visualization and alerting.

## Overview

The monitoring service polls multiple LLM endpoints every 10 minutes to measure their response latency. It tests both OpenAI's main API and various Azure OpenAI regional deployments.

## Metrics Emitted to SignOz

### Latency Metrics
- **llm.deployment.latency** (Histogram) - Response latency in milliseconds for each deployment
  - Attributes: `deployment`, `provider`, `region`, `success`
  - Use for: P50, P95, P99 percentiles

- **llm.deployment.latency.current** (Gauge) - Current latency value for each deployment
  - Attributes: `deployment`, `provider`, `region`, `success`
  - Use for: Real-time dashboards

### Success/Failure Metrics
- **llm.deployment.calls.success** (Counter) - Count of successful API calls
  - Attributes: `deployment`, `provider`, `region`
  
- **llm.deployment.calls.failed** (Counter) - Count of failed API calls
  - Attributes: `deployment`, `provider`, `region`, `error`

### Global Statistics
- **llm.deployment.latency.min** (Gauge) - Minimum latency across all deployments
- **llm.deployment.latency.max** (Gauge) - Maximum latency across all deployments
- **llm.deployment.latency.avg** (Gauge) - Average latency across all deployments
- **llm.deployment.success_rate** (Gauge) - Overall success rate percentage

## Deployment on Render

### 1. Prerequisites
- Render account
- SignOz account with access token
- OpenAI API key
- Azure OpenAI API keys/endpoints (if applicable)

### 2. Environment Variables
Configure these in Render's dashboard as secret environment variables:
- `OPENAI_API_KEY` or configure your secret manager
- `SIGNOZ_ACCESS_TOKEN` or configure your secret manager
- Any Azure OpenAI keys if not hardcoded

### 3. Deploy to Render
```bash
# Push to GitHub
git add .
git commit -m "Add LLM latency monitoring"
git push

# In Render Dashboard:
# 1. Create New > Cron Job
# 2. Connect your GitHub repo
# 3. Use the render.yaml configuration
# 4. Add environment variables
# 5. Deploy
```

### 4. Manual Testing
```bash
# Test locally with single run
python poll_llm_deployments.py

# Test continuous mode locally
RUN_MODE=continuous python poll_llm_deployments.py
```

## SignOz Dashboard Setup

### Creating a Dashboard
1. Go to SignOz > Dashboards > New Dashboard
2. Add panels with these queries:

#### Latency Percentiles Panel
```
histogram_quantile(0.95, sum by (region, le) (rate(llm_deployment_latency_bucket[5m])))
```

#### Success Rate Panel
```
llm_deployment_success_rate
```

#### Deployment Status Table
```
llm_deployment_latency_current
```

#### Error Rate by Region
```
sum by (region) (rate(llm_deployment_calls_failed[5m]))
```

### Setting up Alerts
1. Go to SignOz > Alerts > New Alert
2. Example alert conditions:
   - High latency: `llm_deployment_latency_current > 5000` (5 seconds)
   - Low success rate: `llm_deployment_success_rate < 90`
   - Deployment failures: `rate(llm_deployment_calls_failed[5m]) > 0.1`

## Monitoring Best Practices

1. **Cost Management**: The service uses minimal tokens (1 token response to "Hi") to minimize costs
2. **Timeout Settings**: 10-second timeout prevents hanging requests
3. **Regional Coverage**: Monitors multiple Azure regions to identify the best performing endpoints
4. **Error Tracking**: Failed requests are tracked with error details for debugging

## Troubleshooting

### No metrics appearing in SignOz
1. Check Render logs for errors
2. Verify SignOz access token is correct
3. Ensure network connectivity from Render to SignOz

### High failure rates
1. Check API keys are valid
2. Verify Azure endpoints are accessible
3. Check for rate limiting on API keys

### Missing deployments
1. Update the `azure_endpoints` dictionary in `poll_llm_deployments.py`
2. Ensure API keys are configured for new endpoints