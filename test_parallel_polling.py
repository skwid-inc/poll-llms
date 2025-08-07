#!/usr/bin/env python3
"""
Test script for the parallel polling functionality.
This script runs a quick test of the parallel polling with fewer requests.
"""

import asyncio
import os
import sys

# Set environment variables for testing
os.environ["USE_PARALLEL"] = "true"
os.environ["NUM_REQUESTS_PER_DEPLOYMENT"] = "3"  # Use only 3 requests for quick testing
os.environ["RUN_MODE"] = "single"

# Import and run the main function
from long_poll_llm_deployments import main

if __name__ == "__main__":
    print("Running parallel polling test with 3 requests per deployment...")
    print("This is a test run with reduced requests to verify functionality.")
    print("-" * 80)
    
    try:
        asyncio.run(main())
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        sys.exit(1)