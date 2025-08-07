"""
Placeholder AppConfig for the LLM monitoring service.
This provides the minimal configuration needed for otel.py to work.
"""

class AppConfig:
    """Configuration class for the application"""
    
    def __init__(self):
        self.service_name = "llm-latency-monitor"
        self.client_name = "llm-monitor"
        self.call_metadata = {}
    
    def get_call_metadata(self):
        """Get call metadata"""
        return self.call_metadata