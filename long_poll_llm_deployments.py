import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json
import os
import uuid
import aiohttp
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
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

# Optional import for direct OpenAI SDK
try:
    from openai import AsyncOpenAI
except Exception as e:
    AsyncOpenAI = None  # type: ignore
    logger.warning(f"OpenAI SDK not available: {e}")

FAT_PROMPT = """
### **Tools**
- authenticate_customer: Used to verify customer identity with appropriate parameters:
  - confirmed_name: True/False based on identity confirmation
  - third_party_caller: True if someone else is on the phone on behalf of John Thompson
  - unclear_identity_requesting_information: True if caller asks about call purpose
  - unclear_identity_requesting_identification: True if caller asks who you are
  - unclear_identity_requesting_origin: True if caller asks about call origin
  - unclear_who_agent_is_seeking: True if caller asks who you're looking for
  - customer_not_available: True if someone answers but says customer is not available, not home, or cannot come to phone
  - wrong_number_status: One of
      • unverified_wrong_number  – caller says it's the wrong number without confirming their identity
      • verified_wrong_number    – caller gives a clear confirmation of their identity AND says it's the wrong number


- handle_compliance_or_transfer_request: Used for ANY of the following compliance scenarios:
  - Customer disputes the validity of the charge -> i_dont_owe_this_charge
  - Military service or SCRA accommodations -> scra_military
  - Disability or ADA accommodation requests -> ada_accommodation
  - Bankruptcy mentions -> bankruptcy
  - Attorney representation or legal action mentions -> attorney_representation
  - Account holder is deceased (CRITICAL: Only when the actual account holder has passed away, NOT when customer mentions attending a funeral or someone else dying) -> deceased
  - Stop calls requests ("stop calling me", "don't call me anymore", "remove me from your call list", "cease all calls") -> stop_calling
  - Do not call THIS SPECIFIC number only ("don't call my work number", "don't call me at this number", "stop calling this number", "don't use this number") -> do_not_call_number
  - Customer wants to UPDATE their number or be called at a different number ("call me at...", "my new number is...", "use my other number", "here's my cell") -> different_number
  - Stop email requests (complete cessation WITHOUT TCPA mentions: "stop emailing me", "don't email me", "do not email", "do not send me collections emails") -> stop_emails
  - Stop letter/mail requests (complete cessation WITHOUT TCPA mentions: "stop sending me letters", "don't mail me", "no more letters", "stop sending mail") -> stop_mail
  - Complaints about call frequency ("you call me too much", "calling every day", "you're harassing me", "stop bothering me", "constant calls", any mention of "TCPA") -> too_many_calls
  - Customer doesn't want to talk to AI/bots ("I don't want to talk to a bot", "No AI") -> does_not_want_ai
  - Permanent time restrictions about WHEN to call, not which number ("don't call before 4pm", "no work hours", "not during dinner", "no weekends", "only call after 5pm") -> permanent_inconvenient_time
  - Spanish language service needed -> spanish_speaker
  - Other language service needed (not English/Spanish) -> other_language
  - Relay operator services needed -> operator_relay
  - Spouse or partner handles this account -> spouse_handles_bills
  - Severe hardship that PREVENTS payment: job loss, ongoing medical emergency affecting ability to pay, natural disaster impact, death in family causing financial hardship (NOTE: Do NOT trigger for past events like "I was in the hospital last month" or "I attended a funeral" if customer is now able to discuss payment) -> hardship
  - Customer attempts to hand off call to a third party ("let me give the phone to someone else", "I'll put my friend on", "here talk to my spouse") -> third_party_handoff
  - Profanity or hostile language -> hostile_customer
  - Intent to file a complaint -> complaint
  - Credit reporting, credit score, or credit impact questions ("negative credit impact", "affect my credit", "credit score", "credit report") -> credit_reporting
  - Questions about consequences of non-payment -> asks_consequences_no_payment
  - Extreme frustration with strong emotional language ("absolutely furious", "can't take it anymore") -> extreme_frustration
  - Self-harm threats including self-neglect ("I'll starve") -> self_harm
  - Customer wants to add a new bank account or payment method -> add_funding_account
  - CLEAR temporary unavailability: Customer definitively states they need to end this specific call ("I'll call back", "I have to go", "let me call you back", "I need to go right now") -> temporary_inconvenient_time
  - AMBIGUOUS timing mentions: Any unclear availability statement where you cannot determine if they want to: (a) end this call, (b) set permanent restrictions, or (c) continue despite inconvenience ("I'm busy", "I'm at work", "bad time", "not a good time", "eating dinner", "don't call me at work" - unclear if work number or work hours) -> ambiguous_timing_request
  - Customer cannot hear you or the connection is too poor to continue (for example, if the customer mentions repeated phrases like "I can't hear you", "are you there?", "bad connection", "you're breaking up", or if there are multiple requests to repeat) -> bad_connection
  - Customer explicitly requests human agent or live person -> transfer
  - Payment plan or arrangement requests -> transfer
  - Explicit due date extension requests (e.g., "can I get an extension", "I need more time") -> transfer
  - Fee waiver requests -> transfer
  - Account modification requests -> transfer
  - Payment application or posting questions -> transfer
  - Autopay setup or issues -> transfer
  - Address or contact changes -> transfer
  - Fraud or dispute claims -> transfer
  - Technical issues with online account -> transfer
  - Payment processing questions -> transfer
  - Payment history or statement requests -> transfer

  NOTE: Do NOT call for:
  - Simple account questions
  - Normal payment negotiations
  - Mild frustration
  - Clarification requests
  - Past medical/life events if customer is currently able to make payment
  - Mentions of funerals/hospital visits that don't affect current payment ability


### **Tool Usage Rules**
- **PRIORITY ORDER**: ALWAYS check for compliance issues FIRST before any other tool usage. Any compliance scenario = immediate handle_compliance_or_transfer_request with appropriate compliance_request_type
- When calling ANY tool, output ONLY the tool call - NO accompanying text, explanations, or transitions.
- CRITICAL: ALWAYS use appropriate tools before choosing to end the call. The safety fallback "If you have any questions or need further assistance, please feel free to reach out to us. Have a great day. Goodbye!" should ONLY be used if no tool applies.
- When a customer STATES they're ending the call (e.g., "I'll call back"), use handle_compliance_or_transfer_request(compliance_request_type=temporary_inconvenient_time).
- CRITICAL: Questions about calling back (e.g., "can I call back?", "can I call back later?") require handle_compliance_or_transfer_request(compliance_request_type=ambiguous_timing_request), NOT temporary_inconvenient_time.

**NEVER**:
- Generate text like "Let me check that for you" before tool calls
- Explain what you're doing with the tool
- Summarize tool results in your own words
- Just respond with the tool call and nothing else

### **Agent Identity & Tone**
- Identity: English speaking virtual assistant at Chase Bank
- Tone: Professional, clear, and helpful
- Conversation Guidelines: Speak naturally and conversationally, like a real person
- Do NOT use Markdown formatting such as asterisks, underscores, or hashtags for emphasis. Use plain text only.

### **Primary Task & Scope**
#### **Main Objective**
Verify the identity of the customer on the phone by confirming they are John Thompson.

#### **Scope Restrictions**
Focus on identity verification and handling compliance scenarios during authentication. Do not discuss account details until identity is confirmed. Today is the sixth of August, 2025.

### **Process Workflow**

#### **CRITICAL GUIDELINES**
- **IMMEDIATE COMPLIANCE PRIORITY**: If customer exhibits ANY compliance-related behavior or mentions ANY compliance topic (hostile language, military service, disability, bankruptcy, stop calling, language needs, etc.), IMMEDIATELY call handle_compliance_or_transfer_request with the appropriate compliance_request_type - DO NOT attempt authentication or ask clarifying questions. Compliance ALWAYS takes precedence over authentication.
- CRITICAL: If customer mentions ANY compliance topic (military, disability, bankruptcy, stop calling, etc.), IMMEDIATELY call handle_compliance_or_transfer_request - do NOT continue with authentication
- NEVER assume you know who you're speaking with until they explicitly confirm or deny being John Thompson
- Always get a clear confirmation response to the question "Am I speaking with John Thompson?" before proceeding with any authentication
- If someone returns to the call after being put on hold, ALWAYS ask again "To verify, am I speaking with John Thompson?" regardless of previous conversation context
- If someone identifies as a different person (like a relative), ALWAYS ask them to put John Thompson on the phone
- When someone new comes to the phone, ALWAYS start the verification process again with "To verify, am I speaking with John Thompson?"
- CRITICAL: Authentication requires CLEAR, FULL NAME confirmation in the customer's MOST RECENT response/turn. Do NOT authenticate based on earlier turns - only the current turn matters. Do NOT authenticate with: (1) Ambiguous responses like "mhmm", "sure", "okay", "alright" - ask "I need to confirm - is this John Thompson?" (2) Partial names like "this is John", "I'm Thompson" - ask "To confirm, is your full name John Thompson?" (3) Any unclear or vague response. ONLY authenticate when customer gives clear affirmative ("yes", "yeah", "correct") to the FULL NAME question IN THEIR MOST RECENT RESPONSE. 
- CRITICAL: If the customer wants to call back later or asks about the call back number, you MUST inform them that the call back number is 18003134150
- CRITICAL: Partial names are NOT sufficient for authentication. If customer only provides first/last name (e.g., "this is John", "I'm Thompson"), you MUST ask "To confirm, is your full name John Thompson?" Do NOT authenticate until you get full name confirmation.
- Handle compliance situations immediately when they arise using the handle_compliance_or_transfer_request tool

#### **Step 1: Verify Customer Identity**
- Begin by asking if you are speaking with John Thompson

**If customer mentions being unavailable (busy, work, dinner, etc.) instead of answering:**
- IMMEDIATELY call handle_compliance_or_transfer_request(compliance_request_type=ambiguous_timing_request)
- The tool will guide you on how to respond naturally

- Listen carefully to the response and determine the appropriate action:
  - If customer provides a confirmation that seems ambiguous, you should call authenticate_customer with ambiguous_confirmation=True. Remember that the customer needs to explicitly say yes to confirm their identity.
  - If customer gives ambiguous response (like "mhmm", "uh-huh", "mmm", "sure", "okay", "alright", or other vague sounds): Ask for clarification by saying "I need to confirm - is this John Thompson?" and wait for a clear yes/no response
  - If customer provides only partial name (first name only or last name only like "this is John", "I'm Thompson"): Ask for full name confirmation by saying "To confirm, is your full name John Thompson?" - If the customer continues to provide partial name, you **MUST** call the authenticate_customer tool with ambiguous_confirmation=True
  - If customer clearly confirms identity with a clear affirmative response (says "yes", "yeah", "correct" etc.): IMMEDIATELY call authenticate_customer tool with confirmed_name=True. DO NOT ask any follow-up questions or additional verification.
  - If customer clearly denies identity (says "no", "wrong person", etc.): Call authenticate_customer tool with confirmed_name=False
  - If customer mentions a name (either their own or someone else's): Carefully check if the name matches John Thompson or the cosigner:
     - If name doesn't match John Thompson: Ask "I need to speak with John Thompson. Is that you?" If they clearly state they are not John Thompson, say "Thank you. I need to speak with John Thompson. Is John Thompson available?"
  - If someone says customer is not available, not home, can't come to phone: Call authenticate_customer with customer_not_available=True
  - If person explicitly states this is wrong number: Call authenticate_customer with wrong_number_status=unverified_wrong_number if they don't confirm their identity. If they confirm their identity, call authenticate_customer with wrong_number_status=verified_wrong_number
  - If third party (not John Thompson) wants to make payment, know payment amount, or discuss account: Call authenticate_customer with third_party_wants_account_discussion=True
  - If customer indicates they're calling on behalf of John Thompson: Call authenticate_customer tool with third_party_caller=True
  - If customer asks about the purpose of the call: Call authenticate_customer with unclear_identity_requesting_information=True
  - If customer asks who you are: Call authenticate_customer with unclear_identity_requesting_identification=True
  - If customer asks about call origin: Call authenticate_customer with unclear_identity_requesting_origin=True
  - If customer asks who you're looking for: Call authenticate_customer with unclear_who_agent_is_seeking=True
  
- CRITICAL: After responding to identification questions like "who is this" or returning from hold, ALWAYS follow up with exactly: "To verify, am I speaking with John Thompson?" and wait for a clear response before proceeding

- If the customer indicates the desired party is available or nearby, say exactly: "Can you please put John Thompson on the phone?"
  - If they say yes (agreeing to get the customer), DO NOT call authenticate_customer tool yet
  - If they decline to put the customer  on the phone, call the authenticate_customer tool with confirmed_name=False

- **Third Party Handling**: When speaking with someone other than John Thompson:
  - If they simply answer and don't express interest in the account: Use third_party_caller=True (ends call)
  - If they want to make a payment, know payment amount, or discuss account details: Use third_party_wants_account_discussion=True (transfers to agent)
  - Listen carefully for phrases like "I can make the payment", "how much is owed", "what's the balance", "I want to pay", or similar account-related inquiries

- **Wrong Number Handling**: 
  - Listen for explicit statements like "wrong number", "you have the wrong number", "nobody by that name here", "this isn't their number", "yes but you have the wrong number", "this is [customer_full_name] but you have the wrong number"
  - When identified as wrong number: Use wrong_number_status=unverified_wrong_number if they don't confirm their identity. If they confirm their identity, call authenticate_customer with wrong_number_status=verified_wrong_number (ends call with apology and record update message)

#### **Step 2: Handle Special Situations**
- **Compliance Scenarios**: If the customer mentions any of the following during authentication, use handle_compliance_or_transfer_request:
  - Hostile behavior (insults, profanity, aggressive language) - use compliance_request_type=hostile_customer
  - Military service/SCRA requests
  - Any disability mention or accessibility need (ADHD, hearing issues, etc.)
  - Bankruptcy or attorney representation
  - Requests to stop calling or communication preferences
  - Language assistance needs
  - Hostile behavior or demands for supervisor
  - EXCEPTION: If the customer complains about email or letter frequency (e.g. 'you email me too much', 'you send me too many letters'), acknowledge their complaint but continue with the conversation - do NOT use handle_compliance_or_transfer_request for email complaints
  
- **On-Hold Handling**: If someone says they will get John Thompson for you using phrases like "hold on", "please hold", "one second", "give me a minute", "wait", or any similar expression:
  - FIRST call set_hold_timeout tool to extend the silence timeout
  - THEN respond with: "Thank you, I'll wait."
  - When someone returns to the phone after being on hold, ALWAYS say exactly: "To verify, am I speaking with John Thompson?" Do not call any tools until you get a clear response to this question
- **Co-signer On-Hold Handling**: If someone says they will get either the primary borrower or cosigner on the phone:
  - FIRST call set_hold_timeout tool to extend the silence timeout
  - Ask for clarification: "Thank you. Could you please confirm whether I'll be speaking with John Thompson or None?". Call set_hold_timeout tool again to wait after this turn
  - After they specify which person, respond with: "Thank you, I'll wait for [the name they specified]."
  - When someone returns to the phone, ALWAYS verify identity with: "To verify, am I speaking with [the name they specified]?"
- **General Hold/Wait Requests**: If the person needs time for any other reason during authentication (e.g., "let me close my door", "my baby is crying", "I need to pull over", "give me a moment"):
  - FIRST call set_hold_timeout tool
  - THEN acknowledge appropriately (e.g., "Of course, take your time" or "No problem, I'll wait")
  - When they return, continue with authentication where you left off
- Do not ask to put John Thompson on the phone again if you've been told they are unavailable
- Listen carefully for phrases like "they can't come to the phone", "they're not here", "they're unavailable", or similar statements indicating the customer cannot be reached
- If someone clearly states that John Thompson cannot come to the phone, is unavailable, or is not present, call the authenticate_customer tool with customer_not_available=True (not confirmed_name=False)
- If customer mentions driving, acknowledge safety concerns and then continue with the authentication process by repeating: "To verify, am I speaking with John Thompson?"

### **Tool Usage Summary**
- authenticate_customer: Call with the appropriate parameters based on customer's response
- handle_compliance_or_transfer_request: Call when customer mentions compliance-related scenarios or requests transfer
- handle_off_topic: ALWAYS call this tool when customer asks about ANYTHING unrelated to authentication (weather, sports, personal questions, etc.)
- set_hold_timeout: Call when someone asks you to hold/wait while they get the customer or need time

If the customer authenticates, CALL THE authenticate_customer TOOL RIGHT AWAY.

## Calculation Requests - CRITICAL RULES

**NEVER perform calculations. NEVER state calculated amounts.**

When customers express payment amounts using calculations, percentages, or fractions:
- Do NOT perform any calculations
- Do NOT confirm or suggest calculated amounts
- Do NOT calculate the amount
- Do NOT state what the calculation would equal
- Do NOT say phrases like "that would be" followed by an amount
- Always request the specific dollar amount they can pay

Examples of what NOT to do:
- Customer: "half" → DO NOT say "That would be $821.31"
- Customer: "25%" → DO NOT calculate or state 25% of any amount
- Customer: "a third" → DO NOT calculate or state what a third equals

Instead, always respond with variations of:
- "I need the specific dollar amount you can pay"
- "What exact dollar amount works for you?"
- "Please tell me the exact amount you'd like to pay"

If customer persists with calculation-based requests after 3 attempts, offer to transfer to someone who can help.

## Credit Reporting, Interest & Financial Consequences - CRITICAL RULES

**NEVER mention credit impacts, interest charges, or financial consequences beyond late fees.**

Prohibited topics:
- Credit bureau reporting or credit scores
- Interest charges, rates, or accrual
- Phrases like "affect your credit" or "interest will continue to accrue" or "good standing"
- Any financial implications except late fees

When customers ask about credit impacts or credit scores:
- IMMEDIATELY call handle_compliance_or_transfer_request(compliance_request_type=credit_reporting)
- Do NOT attempt to answer or deflect these questions

When customers ask why they should pay or about other financial consequences:
- Use: "I'm here to discuss payment options for your account"
- Use: "Let's focus on finding a payment solution that works for you"

Remember: You only know about payment collection. Focus on collecting exact payment amounts and dates.
"""

# Tool Schemas
class AuthenticationSchema(BaseModel):
    confirmed_name: Optional[bool] = Field(
        default=False,
        description=(
            "Set to True ONLY if the customer clearly and unambiguously confirms their FULL identity in the most-recent turn.\n"
            "Valid confirmations include: 'Yes', 'That's me', 'Yes I am', 'Yes you are', 'Speaking', or an explicit statement that "
            "includes BOTH first and last name, e.g. 'This is [Customer First Name] [Customer Last Name]'.\n"
            "DO NOT authenticate when the customer provides only a first name ('this is [Customer First Name]') or only a last name "
            "('I'm [Customer Last Name]') or any other partial / ambiguous response."
        ),
    )
    third_party_caller: Optional[bool] = Field(
        default=None,
        description="True if the customer is a third party caller.",
    )
    unclear_identity_requesting_information: Optional[bool] = Field(
        default=None,
        description="True if the customer asks a question regarding the purpose of the call.",
    )
    unclear_identity_requesting_identification: Optional[bool] = Field(
        default=None,
        description="True if the customer asks a question regarding the identity of the assistant.",
    )
    unclear_identity_requesting_origin: Optional[bool] = Field(
        default=None,
        description="True if the customer asks a question regarding the origin of the call.",
    )
    unclear_who_agent_is_seeking: Optional[bool] = Field(
        default=None,
        description="True if the customer asks who the agent is trying to reach or looking for (e.g., 'who are you looking for?', 'who do you need to speak with?', 'who do you want?').",
    )
    is_cosigner: Optional[bool] = Field(
        default=None,
        description="True if the person you're speaking with is the cosigner for the account.",
    )
    customer_not_available: Optional[bool] = Field(
        default=None,
        description="True if someone answers but states the customer is not available, not home, or cannot come to the phone.",
    )
    wrong_number_status: Optional[str] = Field(
        default=None,
        description=(
            "Indicates the caller claims we have the wrong number.\n"
            "- unverified_wrong_number  → caller says it's the wrong number without confirming their identity (e.g. 'This is the wrong number', 'You have the wrong number', 'You are calling the wrong number').\n"
            "- verified_wrong_number    → caller gives a clear confirmation of their identity AND says it's the wrong number (e.g. \"Yes, this is Brad Thompson but you are calling the wrong number\", 'Yes but you are calling the wrong number', 'Yes but you are calling the wrong number').\n"
        ),
    )
    third_party_wants_account_discussion: Optional[bool] = Field(
        default=None,
        description="True if a third party (someone other than the primary customer or cosigner) wants to make a payment, know the payment amount, or discuss the account.",
    )
    third_party_inquiry_when_unavailable: Optional[bool] = Field(
        default=None,
        description="True if a third party asks questions about the call when the customer is not available.",
    )
    ambiguous_confirmation: Optional[bool] = Field(
        default=None,
        description="True if the customer's response is ambiguous and does not clearly confirm or deny their identity. Examples of this can include 'yup', 'yep', 'yeah I guess', 'ye sure', 'yes i guess', 'most probably yes' AND partial names like 'this is [Customer First Name]', 'I'm [Customer Last Name]'",
    )
    unclear_response_needs_clarification: Optional[bool] = Field(
        default=None,
        description="True ONLY when the customer's response appears to be a transcription error where real words were captured but they don't make semantic sense in the context of the authentication question. This is NOT for off-topic responses (which have clear meaning but are unrelated to authentication). Use this when the response contains words that seem misheard or form nonsensical phrases in context. For example, medical terms appearing randomly, words that sound similar to expected responses but make no sense, or grammatically correct sentences with no logical meaning in this context.",
    )

class ComplianceTransferSchema(BaseModel):
    compliance_request_type: str = Field(
        description="Type of compliance or transfer request"
    )

class OffTopicSchema(BaseModel):
    pass

class HoldTimeoutSchema(BaseModel):
    pass

# Tool Definitions
@tool(args_schema=AuthenticationSchema)
async def authenticate_customer(**args):
    """Call this tool to authenticate the customer."""
    return {"status": "authenticated", "args": args}

@tool(args_schema=ComplianceTransferSchema)
async def handle_compliance_or_transfer_request(**args):
    """Handle compliance scenarios or transfer requests."""
    return {"status": "compliance_handled", "args": args}

@tool(args_schema=OffTopicSchema)
async def handle_off_topic(**args):
    """Handle off-topic conversations."""
    return {"status": "off_topic_handled", "args": args}

@tool(args_schema=HoldTimeoutSchema)
async def set_hold_timeout(**args):
    """Set hold timeout when customer asks to wait."""
    return {"status": "hold_timeout_set", "args": args}

def get_auth_tools():
    """Get all authentication tools."""
    return [
        authenticate_customer,
        handle_compliance_or_transfer_request,
        handle_off_topic,
        set_hold_timeout,
    ]

def convert_langchain_tools_to_openai(tools):
    """Convert LangChain tools to OpenAI SDK format."""
    openai_tools = []
    for tool in tools:
        # Extract the schema from the tool
        if hasattr(tool, 'args_schema'):
            schema = tool.args_schema.schema()
            properties = schema.get('properties', {})
            required = schema.get('required', [])
        else:
            properties = {}
            required = []
        
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
        openai_tools.append(openai_tool)
    return openai_tools

def create_chat_prompt_template(system_prompt: str, user_prompt: Optional[str] = None) -> ChatPromptTemplate:
    """Create a chat prompt template for authentication."""
    if user_prompt:
        system_prompt += f"\n{user_prompt}"
    
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])

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
    
@dataclass
class LatencyResult:
    """Result from a latency test"""
    deployment: str
    provider: str
    latency_ms: float  # Total time to complete response
    timestamp: datetime
    success: bool
    error: Optional[str] = None
    tokens_used: int = 0
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
            ('AUSTRALIAEAST', 'gpt-4o', os.getenv('AZURE_OPENAI_ENDPOINT_AUSTRALIAEAST'), os.getenv('AZURE_OPENAI_API_KEY_AUSTRALIAEAST')),
            ('AUSTRALIAEAST-mini', 'gpt-4o-mini', os.getenv('AZURE_OPENAI_ENDPOINT_AUSTRALIAEAST_MINI'), os.getenv('AZURE_OPENAI_API_KEY_AUSTRALIAEAST')),
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
                 azure_api_version: str = "2024-08-01-preview",  # 2024-11-20
                 test_prompt: str = FAT_PROMPT,
                 max_tokens: int = 1,
                 timeout: float = 10.0,
                 use_tools: bool = True,
                 streaming: bool = True,
                 reuse_clients: bool = True):
        """
        Initialize the poller with API keys and configuration
        
        Args:
            openai_api_key: OpenAI API key
            model_name: Model to test (default: gpt-4o)
            azure_api_version: Azure API version
            test_prompt: Simple prompt for testing (default: FAT_PROMPT)
            max_tokens: Max tokens for response (default: 1 for minimal cost)
            timeout: Timeout for each request in seconds
            use_tools: Whether to bind tools to the LLM (default: True)
            streaming: If True enable streaming; note Azure allows only 2 concurrent streaming requests per deployment.
            reuse_clients: If True reuse underlying HTTP client sessions across requests to avoid TLS handshake overhead.
        """
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.azure_api_version = azure_api_version
        self.test_prompt = test_prompt
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.use_tools = use_tools
        self.results_history: List[LatencyResult] = []
        
        # Connection/session handling
        self.streaming = streaming
        self.reuse_clients = reuse_clients
        self._openai_client: Optional[ChatOpenAI] = None
        self._azure_clients: Dict[str, AzureChatOpenAI] = {}
        self._openai_sdk_client: Optional[Any] = None  # Direct SDK client
        
        # Load Azure endpoints on initialization
        self._load_azure_endpoints()
        
        # Initialize direct SDK client if available
        if AsyncOpenAI is not None:
            self._openai_sdk_client = AsyncOpenAI(
                api_key=self.openai_api_key,
                timeout=self.timeout,
                max_retries=0  # No retries for accurate timing
            )
        
    def _create_openai_client(self, model: Optional[str] = None) -> ChatOpenAI:
        """Create OpenAI client"""
        client = ChatOpenAI(
            api_key=self.openai_api_key,
            model=model or self.model_name,
            temperature=0,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=0,  # No retries for accurate timing
            streaming=self.streaming  # Configurable streaming
        )
        
        # Bind tools if enabled
        if self.use_tools:
            client = client.bind_tools(get_auth_tools(), parallel_tool_calls=False)
        
        return client
        
    def _create_azure_client(self, endpoint: str, model: Optional[str] = None) -> AzureChatOpenAI:
        """Create Azure OpenAI client for a specific endpoint"""
        # Use the model from config if provided, otherwise use the model from endpoint config
        endpoint_config = self.azure_endpoints.get(endpoint, {})
        deployment_model = model or endpoint_config.get('model', self.model_name)
        api_key = endpoint_config.get('api_key') if isinstance(endpoint_config, dict) else endpoint_config
        
        client = AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=deployment_model,
            api_version=self.azure_api_version,
            api_key=api_key,
            temperature=0,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=0,
            streaming=self.streaming  # Configurable streaming
        )
        
        # Bind tools if enabled
        if self.use_tools:
            client = client.bind_tools(get_auth_tools(), parallel_tool_calls=False)
        
        return client

    # ------------------------------------------------------------------
    # Cached client helpers to avoid repeated TLS handshakes
    # ------------------------------------------------------------------
    def _get_openai_client(self, model: Optional[str] = None) -> ChatOpenAI:
        """Return a cached OpenAI client (or create one)."""
        # For OpenAI, we create new clients for different models
        if model and model != self.model_name:
            return self._create_openai_client(model)
        
        if self.reuse_clients and self._openai_client is not None:
            return self._openai_client
        client = self._create_openai_client(model)
        if self.reuse_clients:
            self._openai_client = client
        return client

    def _get_azure_client(self, endpoint: str, model: Optional[str] = None) -> AzureChatOpenAI:
        """Return a cached Azure client for the given endpoint (or create one)."""
        if self.reuse_clients and endpoint in self._azure_clients:
            return self._azure_clients[endpoint]
        client = self._create_azure_client(endpoint, model)
        if self.reuse_clients:
            self._azure_clients[endpoint] = client
        return client

    async def _test_deployment_multiple(self, config: DeploymentConfig, num_requests: int = 10) -> List[LatencyResult]:
        """
        Test a single deployment with multiple parallel requests
        
        Args:
            config: Deployment configuration
            num_requests: Number of parallel requests to make
            
        Returns:
            List of LatencyResult objects
        """
        # Run *all* requested trials but throttle streaming concurrency to 2
        max_concurrency = 2 if self.streaming else num_requests

        semaphore = asyncio.Semaphore(max_concurrency)

        async def limited_test() -> LatencyResult | Exception:
            # Acquire permit so that at most `max_concurrency` are inflight
            async with semaphore:
                return await self._test_deployment(config)

        tasks = [asyncio.create_task(limited_test()) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        processed_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(LatencyResult(
                    deployment=config.name,
                    provider=config.provider,
                    latency_ms=0,
                    timestamp=datetime.now(),
                    success=False,
                    error=str(result),
                    tokens_used=0,
                    time_to_first_token_ms=None,
                    completion_time_ms=None,
                    model=config.model
                ))
            else:
                processed_results.append(result)
        
        return processed_results
        
    async def _test_deployment(self, 
                              deployment_config: DeploymentConfig) -> LatencyResult:
        """
        Test a single deployment and measure latency with streaming
        
        Args:
            deployment_config: Configuration for the deployment to test
            
        Returns:
            LatencyResult with timing information including TTFT
        """
        timestamp = datetime.now()
        start_time = None
        first_token_time = None
        full_response = ""
        
        try:
            if deployment_config.provider == 'openai' and deployment_config.use_direct_sdk and self._openai_sdk_client:
                # Use direct OpenAI SDK
                system_prompt = "You are Taylor, a virtual assistant for Chase Bank. You are verifying the identity of the customer on the phone."
                full_prompt = system_prompt + "\n" + str(uuid.uuid4()) + str(uuid.uuid4()) + str(uuid.uuid4()) + self.test_prompt
                
                messages = [
                    {"role": "system", "content": full_prompt},
                    {"role": "user", "content": "Hi, who is this?"}
                ]
                
                if self.streaming:
                    # Begin timing just before sending the request
                    start_time = time.perf_counter()
                    
                    # Stream the response using direct SDK
                    stream = await self._openai_sdk_client.chat.completions.create(
                        model=deployment_config.model or self.model_name,
                        messages=messages,
                        temperature=0,
                        max_tokens=self.max_tokens,
                        stream=True,
                        tools=convert_langchain_tools_to_openai(get_auth_tools()) if self.use_tools else None,
                        parallel_tool_calls=False if self.use_tools else None
                    )
                    
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta:
                            delta = chunk.choices[0].delta
                            has_content = delta.content is not None
                            has_tool_calls = hasattr(delta, 'tool_calls') and delta.tool_calls
                            
                            if first_token_time is None and (has_content or has_tool_calls):
                                first_token_time = time.perf_counter()
                            
                            if has_content:
                                full_response += delta.content
                else:
                    # Non-streaming SDK request
                    start_time = time.perf_counter()
                    response = await self._openai_sdk_client.chat.completions.create(
                        model=deployment_config.model or self.model_name,
                        messages=messages,
                        temperature=0,
                        max_tokens=self.max_tokens,
                        stream=False,
                        tools=convert_langchain_tools_to_openai(get_auth_tools()) if self.use_tools else None,
                        parallel_tool_calls=False if self.use_tools else None
                    )
                    full_response = response.choices[0].message.content or ""
                    first_token_time = start_time
            
            else:
                # Use LangChain clients
                if deployment_config.provider == 'openai':
                    client = self._get_openai_client(model=deployment_config.model)
                else:  # azure
                    client = self._get_azure_client(deployment_config.endpoint, model=deployment_config.model)
                
                # Create messages using the FAT_PROMPT as system message
                # Split the FAT_PROMPT to get system and user parts
                system_prompt = "You are Taylor, a virtual assistant for Chase Bank. You are verifying the identity of the customer on the phone."
                
                # Create the prompt template like production
                prompt_template = create_chat_prompt_template(system_prompt, str(uuid.uuid4()) + str(uuid.uuid4()) +  str(uuid.uuid4()) + self.test_prompt)
                
                # Format messages with the prompt template (system + FAT prompt)
                messages = prompt_template.format_messages(
                    messages=[HumanMessage(content="Hi, who is this?")]
                )
                
                if self.streaming:
                    # Begin timing just before sending the request
                    start_time = time.perf_counter()

                    # Stream the response to measure TTFT
                    async for chunk in client.astream(messages):
                        # Check if chunk has actual content or tool calls
                        has_content = hasattr(chunk, 'content') and chunk.content
                        has_tool_calls = hasattr(chunk, 'tool_calls') and chunk.tool_calls
                        
                        # Set first token time only when we see actual payload
                        if first_token_time is None and (has_content or has_tool_calls):
                            first_token_time = time.perf_counter()
                        
                        # Build response
                        if has_content:
                            full_response += chunk.content
                else:
                    # Non-streaming request (e.g., when running high concurrency)
                    start_time = time.perf_counter()
                    response = await client.invoke(messages)
                    # LangChain returns a ChatMessage; extract content if present
                    full_response = getattr(response, 'content', str(response))
                    first_token_time = start_time
            
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
            
            # Estimate total tokens used for cost tracking (prompt only)
            tokens_used = sum(
                len(getattr(m, "content", "").split())
                for m in messages
                if hasattr(m, "content")
            )
            
            return LatencyResult(
                deployment=deployment_config.name,
                provider=deployment_config.provider,
                latency_ms=total_latency_ms,
                timestamp=timestamp,
                success=True,
                tokens_used=tokens_used,
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
        
        # Direct SDK versions (if available)
        if self._openai_sdk_client:
            deployments.append(
                DeploymentConfig(
                    name="openai-sdk-gpt-4o",
                    provider="openai",
                    model="gpt-4o",
                    use_direct_sdk=True
                )
            )
            deployments.append(
                DeploymentConfig(
                    name="openai-sdk-gpt-4o-mini",
                    provider="openai",
                    model="gpt-4o-mini",
                    use_direct_sdk=True
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
        
        # Test all deployments concurrently
        logger.info(f"Testing {len(deployments)} deployments...")
        tasks = [self._test_deployment(config) for config in deployments]
        results = await asyncio.gather(*tasks)
        
        # Store results in history
        self.results_history.extend(results)
        
        return results
    
    async def poll_all_deployments_parallel(self, num_requests_per_deployment: int = 10) -> Dict[str, List[LatencyResult]]:
        """
        Poll all configured deployments with multiple parallel requests per deployment
        
        Args:
            num_requests_per_deployment: Number of parallel requests per deployment
            
        Returns:
            Dictionary mapping deployment name to list of results
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
        
        # Direct SDK versions (if available)
        if self._openai_sdk_client:
            deployments.append(
                DeploymentConfig(
                    name="openai-sdk-gpt-4o",
                    provider="openai",
                    model="gpt-4o",
                    use_direct_sdk=True
                )
            )
            deployments.append(
                DeploymentConfig(
                    name="openai-sdk-gpt-4o-mini",
                    provider="openai",
                    model="gpt-4o-mini",
                    use_direct_sdk=True
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
        
        # Test all deployments concurrently with multiple requests each
        logger.info(f"Testing {len(deployments)} deployments with {num_requests_per_deployment} requests each...")
        tasks = [self._test_deployment_multiple(config, num_requests_per_deployment) for config in deployments]
        all_results = await asyncio.gather(*tasks)
        
        # Create a dictionary mapping deployment names to their results
        results_by_deployment = {}
        for deployment, results in zip(deployments, all_results):
            results_by_deployment[deployment.name] = results
            # Store results in history
            self.results_history.extend(results)
        
        return results_by_deployment
    
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
            print("\nOpenAI Deployments:")
            print("-"*100)
            print(f"{'Deployment':30s} {'Total':>10s} {'TTFT':>10s} {'Completion':>12s} {'Status':>8s}")
            print("-"*100)
            for r in openai_results:
                status = "✓" if r.success else "✗"
                if r.success and r.time_to_first_token_ms:
                    print(f"{r.deployment:30s} {r.latency_ms:10.2f} {r.time_to_first_token_ms:10.2f} {r.completion_time_ms:12.2f} {status:>8s}")
                else:
                    print(f"{r.deployment:30s} {r.latency_ms:10.2f} {'N/A':>10s} {'N/A':>12s} {status:>8s}")
        
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
            print(f"\n  Total Tokens Used: ~{sum(r.tokens_used for r in results)}")
    
    def print_parallel_results_summary(self, results_by_deployment: Dict[str, List[LatencyResult]]):
        """Print a formatted summary of parallel test results with averages"""
        print("\n" + "="*140)
        print(f"Parallel Latency Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*140)
        
        # Prepare data for summary table
        summary_data = []
        
        for deployment_name, results in sorted(results_by_deployment.items()):
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            if successful_results:
                latencies = [r.latency_ms for r in successful_results]
                ttfts = [r.time_to_first_token_ms for r in successful_results if r.time_to_first_token_ms]
                
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                
                avg_ttft = sum(ttfts) / len(ttfts) if ttfts else None
                min_ttft = min(ttfts) if ttfts else None
                max_ttft = max(ttfts) if ttfts else None
                
                success_rate = len(successful_results) / len(results) * 100
                
                summary_data.append({
                    'deployment': deployment_name,
                    'provider': results[0].provider,
                    'success_rate': success_rate,
                    'avg_latency': avg_latency,
                    'min_latency': min_latency,
                    'max_latency': max_latency,
                    'avg_ttft': avg_ttft,
                    'min_ttft': min_ttft,
                    'max_ttft': max_ttft,
                    'successful': len(successful_results),
                    'failed': len(failed_results),
                    'total': len(results)
                })
            else:
                summary_data.append({
                    'deployment': deployment_name,
                    'provider': results[0].provider if results else 'unknown',
                    'success_rate': 0,
                    'avg_latency': None,
                    'min_latency': None,
                    'max_latency': None,
                    'avg_ttft': None,
                    'min_ttft': None,
                    'max_ttft': None,
                    'successful': 0,
                    'failed': len(results),
                    'total': len(results)
                })
        
        # Print summary by provider
        openai_data = [d for d in summary_data if d['provider'] == 'openai']
        azure_data = [d for d in summary_data if d['provider'] == 'azure']
        
        # Print OpenAI results
        if openai_data:
            print("\nOpenAI Deployments:")
            print("-"*140)
            print(f"{'Deployment':35s} {'Success':>8s} {'Avg Latency':>12s} {'Min/Max Lat':>20s} {'Avg TTFT':>10s} {'Min/Max TTFT':>20s}")
            print("-"*140)
            for d in openai_data:
                success_str = f"{d['successful']}/{d['total']}"
                latency_str = f"{d['avg_latency']:.2f}" if d['avg_latency'] else "N/A"
                minmax_lat_str = f"{d['min_latency']:.0f}-{d['max_latency']:.0f}" if d['min_latency'] else "N/A"
                ttft_str = f"{d['avg_ttft']:.2f}" if d['avg_ttft'] else "N/A"
                minmax_ttft_str = f"{d['min_ttft']:.0f}-{d['max_ttft']:.0f}" if d['min_ttft'] else "N/A"
                print(f"{d['deployment']:35s} {success_str:>8s} {latency_str:>12s} {minmax_lat_str:>20s} {ttft_str:>10s} {minmax_ttft_str:>20s}")
        
        # Print Azure results sorted by average latency
        if azure_data:
            print("\nAzure Deployments (sorted by avg latency):")
            print("-"*140)
            print(f"{'Deployment':35s} {'Success':>8s} {'Avg Latency':>12s} {'Min/Max Lat':>20s} {'Avg TTFT':>10s} {'Min/Max TTFT':>20s}")
            print("-"*140)
            azure_sorted = sorted(azure_data, key=lambda d: d['avg_latency'] if d['avg_latency'] else float('inf'))
            for d in azure_sorted:
                success_str = f"{d['successful']}/{d['total']}"
                latency_str = f"{d['avg_latency']:.2f}" if d['avg_latency'] else "N/A"
                minmax_lat_str = f"{d['min_latency']:.0f}-{d['max_latency']:.0f}" if d['min_latency'] else "N/A"
                ttft_str = f"{d['avg_ttft']:.2f}" if d['avg_ttft'] else "N/A"
                minmax_ttft_str = f"{d['min_ttft']:.0f}-{d['max_ttft']:.0f}" if d['min_ttft'] else "N/A"
                
                # Shorten deployment name for better display
                deployment_display = d['deployment'].replace('azure-https://', '').replace('.openai.azure.com/', '')
                print(f"{deployment_display:35s} {success_str:>8s} {latency_str:>12s} {minmax_lat_str:>20s} {ttft_str:>10s} {minmax_ttft_str:>20s}")
        
        # Print top performers
        successful_deployments = [d for d in summary_data if d['avg_latency']]
        if successful_deployments:
            print("\n" + "="*80)
            print("🏆 Performance Champions (based on averages):")
            print("-"*80)
            
            # Fastest average latency
            fastest_avg = min(successful_deployments, key=lambda d: d['avg_latency'])
            print(f"Fastest Average Latency: {fastest_avg['deployment']}")
            print(f"   Avg: {fastest_avg['avg_latency']:.2f} ms (range: {fastest_avg['min_latency']:.0f}-{fastest_avg['max_latency']:.0f} ms)")
            
            # Fastest average TTFT
            ttft_deployments = [d for d in successful_deployments if d['avg_ttft']]
            if ttft_deployments:
                fastest_ttft = min(ttft_deployments, key=lambda d: d['avg_ttft'])
                print(f"\nFastest Average TTFT: {fastest_ttft['deployment']}")
                print(f"   Avg: {fastest_ttft['avg_ttft']:.2f} ms (range: {fastest_ttft['min_ttft']:.0f}-{fastest_ttft['max_ttft']:.0f} ms)")
            
        # Print detailed results per deployment if needed
        print("\n" + "="*80)
        print("Detailed Results per Deployment (showing first 3):")
        print("="*80)
        
        for deployment_name, results in sorted(results_by_deployment.items())[:3]:  # Show first 3 deployments
            print(f"\n{deployment_name}:")
            print("-"*60)
            for i, r in enumerate(results[:5]):  # Show first 5 results
                if r.success:
                    print(f"  Request {i+1}: {r.latency_ms:8.2f} ms | TTFT: {r.time_to_first_token_ms:8.2f} ms" if r.time_to_first_token_ms else f"  Request {i+1}: {r.latency_ms:8.2f} ms")
                else:
                    print(f"  Request {i+1}: FAILED - {r.error[:50]}...")
            if len(results) > 5:
                print(f"  ... and {len(results) - 5} more requests")
    
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
            if "sdk" in result.deployment:
                sdk_type = "direct_sdk"
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
                # Report total latency as a histogram (for percentiles) - with .long suffix
                report_metrics(
                    name="llm.deployment.latency.long",
                    instrument_type="histogram",
                    value=result.latency_ms,
                    description="LLM deployment total latency in milliseconds (long prompt with tools)",
                    attributes=attributes
                )
                
                # Report TTFT if available - with .long suffix
                if result.time_to_first_token_ms is not None:
                    report_metrics(
                        name="llm.deployment.ttft.long",
                        instrument_type="histogram",
                        value=result.time_to_first_token_ms,
                        description="LLM deployment time to first token in milliseconds (long prompt with tools)",
                        attributes=attributes
                    )
                    
                    # Report completion time - with .long suffix
                    report_metrics(
                        name="llm.deployment.completion_time.long",
                        instrument_type="histogram",
                        value=result.completion_time_ms,
                        description="LLM deployment completion time (after first token) in milliseconds (long prompt with tools)",
                        attributes=attributes
                    )
                
                # Also report as gauges for current values - with .long suffix
                report_gauge(
                    name="llm.deployment.latency.long.current",
                    value=result.latency_ms,
                    description="Current LLM deployment latency in milliseconds (long prompt with tools)",
                    attributes=attributes
                )
                
                if result.time_to_first_token_ms is not None:
                    report_gauge(
                        name="llm.deployment.ttft.long.current",
                        value=result.time_to_first_token_ms,
                        description="Current LLM deployment TTFT in milliseconds (long prompt with tools)",
                        attributes=attributes
                    )
                
                # Count successful calls - with .long suffix
                inc_counter(
                    name="llm.deployment.calls.long.success",
                    value=1,
                    description="Number of successful LLM deployment calls (long prompt with tools)",
                    attributes=attributes
                )
            else:
                # Count failed calls - with .long suffix
                inc_counter(
                    name="llm.deployment.calls.long.failed",
                    value=1,
                    description="Number of failed LLM deployment calls (long prompt with tools)",
                    attributes={**attributes, "error": result.error or "unknown"}
                )
                
                # Report failure as max latency for monitoring - with .long suffix
                report_gauge(
                    name="llm.deployment.latency.long.current",
                    value=999999,  # High value to indicate failure
                    description="Current LLM deployment latency in milliseconds (long prompt with tools)",
                    attributes=attributes
                )
        
        # Report overall statistics
        successful_results = [r for r in results if r.success]
        if successful_results:
            latencies = [r.latency_ms for r in successful_results]
            ttfts = [r.time_to_first_token_ms for r in successful_results if r.time_to_first_token_ms is not None]
            
            # Total latency stats - with .long suffix
            report_gauge(
                name="llm.deployment.latency.long.min",
                value=min(latencies),
                description="Minimum latency across all deployments (long prompt with tools)",
                attributes={"measurement": "global"}
            )
            
            report_gauge(
                name="llm.deployment.latency.long.max",
                value=max(latencies),
                description="Maximum latency across all deployments (long prompt with tools)",
                attributes={"measurement": "global"}
            )
            
            report_gauge(
                name="llm.deployment.latency.long.avg",
                value=sum(latencies) / len(latencies),
                description="Average latency across all deployments (long prompt with tools)",
                attributes={"measurement": "global"}
            )
            
            # TTFT stats - with .long suffix
            if ttfts:
                report_gauge(
                    name="llm.deployment.ttft.long.min",
                    value=min(ttfts),
                    description="Minimum TTFT across all deployments (long prompt with tools)",
                    attributes={"measurement": "global"}
                )
                
                report_gauge(
                    name="llm.deployment.ttft.long.max",
                    value=max(ttfts),
                    description="Maximum TTFT across all deployments (long prompt with tools)",
                    attributes={"measurement": "global"}
                )
                
                report_gauge(
                    name="llm.deployment.ttft.long.avg",
                    value=sum(ttfts) / len(ttfts),
                    description="Average TTFT across all deployments (long prompt with tools)",
                    attributes={"measurement": "global"}
                )
            
            report_gauge(
                name="llm.deployment.success_rate.long",
                value=(len(successful_results) / len(results)) * 100,
                description="Success rate percentage (long prompt with tools)",
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
                'error': r.error or '',
                'tokens_used': r.tokens_used
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
    
    async def continuous_parallel_polling(self, 
                                         interval_seconds: int = 60,
                                         duration_minutes: Optional[int] = None,
                                         num_requests_per_deployment: int = 10):
        """
        Continuously poll deployments with parallel requests at regular intervals
        
        Args:
            interval_seconds: Seconds between polls
            duration_minutes: Total duration to run (None for infinite)
            num_requests_per_deployment: Number of parallel requests per deployment
        """
        start_time = time.time()
        poll_count = 0
        
        while True:
            poll_count += 1
            logger.info(f"Starting parallel poll #{poll_count}")
            
            # Poll all deployments with parallel requests
            results_by_deployment = await self.poll_all_deployments_parallel(num_requests_per_deployment)
            
            # Flatten results for metrics emission
            all_results = []
            for deployment_results in results_by_deployment.values():
                all_results.extend(deployment_results)
            
            # Emit metrics to SignOz
            self.emit_metrics_to_signoz(all_results)
            
            # Print summary
            self.print_parallel_results_summary(results_by_deployment)
            
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
    
    # Create poller instance
    poller = LLMLatencyPoller(
        openai_api_key=openai_api_key,
        model_name="gpt-4o",
        test_prompt=FAT_PROMPT,  # Use the full auth prompt
        max_tokens=150,  # Enough tokens for tool call response
        timeout=30.0,  # Longer timeout for large prompt
        use_tools=True  # Enable tool binding
    )
    
    # Check if we're running as a cron job or continuous mode
    run_mode = os.getenv("RUN_MODE", "single")  # "single" for cron, "continuous" for testing
    use_parallel = os.getenv("USE_PARALLEL", "true").lower() == "true"  # Default to parallel mode
    num_requests = int(os.getenv("NUM_REQUESTS_PER_DEPLOYMENT", "1"))  # Number of parallel requests
    
    if run_mode == "continuous":
        # Run continuously (for testing)
        if use_parallel:
            await poller.continuous_parallel_polling(
                interval_seconds=600,  # 10 minutes
                duration_minutes=None,  # Run indefinitely
                num_requests_per_deployment=num_requests
            )
        else:
            await poller.continuous_polling(
                interval_seconds=600,  # 10 minutes
                duration_minutes=None  # Run indefinitely
            )
    else:
        # Run once and exit (for cron job)
        if use_parallel:
            logger.info(f"Running parallel latency test across all deployments with {num_requests} requests each...")
            
            # Poll all deployments with parallel requests
            results_by_deployment = await poller.poll_all_deployments_parallel(num_requests)
            
            # Flatten results for metrics
            all_results = []
            for deployment_results in results_by_deployment.values():
                all_results.extend(deployment_results)
            
            # Emit metrics to SignOz
            poller.emit_metrics_to_signoz(all_results)
            
            # Print summary for logs
            poller.print_parallel_results_summary(results_by_deployment)
        else:
            logger.info("Running single latency test across all deployments...")
            
            # Poll all deployments
            results = await poller.poll_all_deployments()
            
            # Emit metrics to SignOz
            poller.emit_metrics_to_signoz(results)
            
            # Print summary for logs
            poller.print_results_summary(results)
        
        # Log completion
        logger.info("Latency test completed and metrics emitted to SignOz")


if __name__ == "__main__":
    asyncio.run(main())