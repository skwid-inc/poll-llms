import asyncio
import json
import logging
import os
import re
import time
from typing import Optional

from openai import AsyncOpenAI, BaseModel, AsyncAzureOpenAI
from supabase.client import create_client

from app_config import AppConfig
from secret_manager import access_secret
from utils.feature_flag.client import FeatureFlag
from dr_helpers import LLM_AGENT_KEY, get_call_primary_secondary_mapping, SECONDARY

# Configure logging
logger = logging.getLogger("basic-agent")
print = logger.info

## Keys ##
# Access API keys from secret manager
OPENAI_API_KEY_REGULAR = access_secret("openai-api-key")
OPENAI_API_KEY_SCALE = access_secret("openai-api-key-scale")
CLIENT_NAME = os.getenv("CLIENT_NAME")

OPENAI_API_KEY = (
    OPENAI_API_KEY_SCALE if AppConfig().is_pilot else OPENAI_API_KEY_REGULAR
)


# Model for tracking conversation summary data
class DefaultSummaryTracker(BaseModel):
    """Summary Tracker."""

    desired_payment_amount: Optional[float]
    desired_payment_date: Optional[str]

    class Config:
        extra = "forbid"  # Prevent additional fields from being added


class AllySummaryTracker(BaseModel):
    """Summary Tracker."""

    desired_payment_amount_1: Optional[float]
    desired_payment_date_1: Optional[str]
    desired_payment_amount_2: Optional[float]
    desired_payment_date_2: Optional[str]
    customer_desired_change_notes: Optional[str]

    class Config:
        extra = "forbid"  # Prevent additional fields from being added


supabase_client = create_client(
    "https://dfdvsmtmyhsqslvcvpcl.supabase.co",
    access_secret("campaigns-supabase-key"),
)


# Singleton class for summarizing conversation transcripts
class Summarizer:
    _instance = None

    # Implement singleton pattern
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Summarizer, cls).__new__(cls)
            cls._instance.summary_prompt = cls._instance.init_summary_prompt()
            cls._instance.SummaryTrackerClass = (
                cls._instance.get_summary_tracker_class()
            )
        return cls._instance

    def get_summary_tracker_class(self):
        """Return the appropriate SummaryTracker class based on client."""
        if AppConfig().client_name == "ally":
            return AllySummaryTracker
        else:
            return DefaultSummaryTracker

    def format_summary(self, summary_tracker):
        """
        Format a SummaryTracker object into a human-readable summary string.

        Args:
            summary_tracker: A DefaultSummaryTracker or AllySummaryTracker object

        Returns:
            A formatted string summarizing the information in the SummaryTracker.
        """
        if isinstance(summary_tracker, AllySummaryTracker):
            return self.format_ally_summary(summary_tracker)
        else:
            return self.format_default_summary(summary_tracker)

    def format_default_summary(
        self, summary_tracker: DefaultSummaryTracker
    ) -> str:
        """Format a DefaultSummaryTracker into a human-readable string."""
        summary_parts = []

        # Add desired payment amount if present
        if summary_tracker.desired_payment_amount:
            summary_parts.append(
                f"\n* Customer has confirmed the following payment amount: ${summary_tracker.desired_payment_amount}"
            )
        else:
            summary_parts.append(
                "\n* Customer has not confirmed payment amount"
            )

        # Add desired payment date if present
        if summary_tracker.desired_payment_date:
            summary_parts.append(
                f"\n* Customer has confirmed the following payment date: {summary_tracker.desired_payment_date}"
            )
        else:
            summary_parts.append("\n* Customer has not confirmed payment date")

        # Join all parts with newlines
        if summary_parts:
            summary = " ".join(summary_parts)
        else:
            return ""  # Return empty string if no summary parts
        return f"Summary of conversation so far:{summary}\nGuidance:\n"

    def format_ally_summary(self, summary_tracker: AllySummaryTracker) -> str:
        """Format an AllySummaryTracker into a human-readable string."""
        summary_parts = []

        # Add first payment details if present
        if summary_tracker.desired_payment_amount_1:
            summary_parts.append(
                f"\n* Customer has confirmed payment 1 amount: ${summary_tracker.desired_payment_amount_1}"
            )
        else:
            summary_parts.append(
                "\n* Customer has not confirmed payment 1 amount"
            )

        if summary_tracker.desired_payment_date_1:
            summary_parts.append(
                f"\n* Customer has confirmed payment 1 date: {summary_tracker.desired_payment_date_1}"
            )
        else:
            summary_parts.append(
                "\n* Customer has not confirmed payment 1 date"
            )

        # Add second payment details if present
        if summary_tracker.desired_payment_amount_2:
            summary_parts.append(
                f"\n* Customer has confirmed payment 2 amount: ${summary_tracker.desired_payment_amount_2}"
            )
        else:
            summary_parts.append(
                "\n* Customer has not confirmed payment 2 amount"
            )

        if summary_tracker.desired_payment_date_2:
            summary_parts.append(
                f"\n* Customer has confirmed payment 2 date: {summary_tracker.desired_payment_date_2}"
            )
        else:
            summary_parts.append(
                "\n* Customer has not confirmed payment 2 date"
            )

        if summary_tracker.customer_desired_change_notes:
            summary_parts.append(
                f"\n* Requested changes: {summary_tracker.customer_desired_change_notes}"
            )

        # Join all parts with newlines
        if summary_parts:
            summary = " ".join(summary_parts)
        else:
            return ""  # Return empty string if no summary parts
        return f"Summary of conversation so far:{summary}\nGuidance:\n"

    def init_summary_prompt(self):
        if CLIENT_NAME == "ally":
            return self.ally_summary_prompt()
        else:
            return self.default_summary_prompt()

    def invoke_summary(self, transcript):
        if AppConfig().feature_flag_client.is_feature_enabled(
            FeatureFlag.SUMMARIZER_ENABLED,
            AppConfig().client_name,
            AppConfig().call_metadata.get("call_id"),
        ):
            summarize_task = asyncio.create_task(
                self.summarize_transcript(transcript)
            )
            logger.info("Started async summarize_transcript task")
            return summarize_task
        return None

    async def wait_for_summarize_task(self, summarize_task, invoke_result):
        if not summarize_task:
            return
        if (
            AppConfig().feature_flag_client.is_feature_enabled(
                FeatureFlag.SUMMARIZER_ENABLED,
                AppConfig().client_name,
                AppConfig().call_metadata.get("call_id"),
            )
            and invoke_result.tool_calls
        ):
            try:
                start_time = time.time()
                await asyncio.wait_for(summarize_task, timeout=1.0)
                elapsed = time.time() - start_time
                logger.info(
                    f"Summarize task waited for an extra {elapsed:.3f} seconds after llm invoke complete"
                )
            except asyncio.TimeoutError:
                logger.error("Summarize task timed out after 1 second")

    async def create_summary(
        self,
        content: str,
        save_to_supabase: bool = False,
    ) -> DefaultSummaryTracker | AllySummaryTracker:
        """
        Create a summary of a conversation transcript using OpenAI's GPT-4 model.

        Args:
            content (str): The conversation transcript to summarize

        Returns:
            str: The formatted summary of the conversation transcript
        """
        # Create the full prompt with system context and transcript
        input_messages = [
            {"role": "system", "content": self.summary_prompt},
            {
                "role": "user",
                "content": content,
            },
        ]
        logger.info(f"Sending to summary llm: {input_messages}")

        # Primary/secondary logic
        use_primary = get_call_primary_secondary_mapping(LLM_AGENT_KEY) != SECONDARY
        if use_primary:
            client = AsyncOpenAI(
                base_url="https://api.openai.com/v1",
                api_key=OPENAI_API_KEY,
            )
            llm_params = {
                "model": "gpt-4o",
                "temperature": 0,
                "messages": input_messages,
                "response_format": self.SummaryTrackerClass,
            }
            if AppConfig().is_pilot:
                llm_params["service_tier"] = "auto"
            response = await client.beta.chat.completions.parse(**llm_params)
        else:
            logger.info("Using Azure OpenAI as fallback")
            client = AsyncAzureOpenAI(
                azure_endpoint="https://azureoaivf.openai.azure.com",
                api_key=access_secret("azure-openai-api-key"),
                api_version="2024-08-01-preview",
            )
            llm_params = {
                "model": "gpt-4o",
                "temperature": 0,
                "messages": input_messages,
                "response_format": self.SummaryTrackerClass,
            }
            response = await client.beta.chat.completions.parse(**llm_params)
        summary_tracker = response.choices[0].message.parsed
        return summary_tracker

    async def summarize_transcript(self, transcript: list):
        """
        Asynchronously summarize a conversation transcript using OpenAI's GPT-4 model.

        Args:
            transcript (list): List of conversation messages to summarize

        Returns:
            None - Updates AppConfig with the summary results
        """
        try:
            if not transcript:
                return

            # logger.info(f"Transcript: {transcript}")

            # Combine the transcript messages into a format the LLM can understand
            messages = []
            for idx, msg in enumerate(transcript):
                # For tool messages, only keep content matching the summary pattern
                if msg.type == "tool":
                    if match := re.search(
                        r"Summary of conversation so far:.*?(?=\nGuidance:)",
                        msg.content,
                        re.DOTALL,
                    ):
                        messages.append(f"{msg.type}: {match.group()}")
                    elif idx == 1:
                        # Include the entry message in the summary transcript
                        messages.append(f"{msg.type}: {msg.content}")
                # Filter out any messages that contain tool calls since they're internal
                elif not hasattr(msg, "tool_calls") or not msg.tool_calls:
                    messages.append(f"{msg.type}: {msg.content}")
            messages_text = "\n".join(messages)
            if not messages_text:
                return

            content = f"Here's the conversation transcript to summarize:\n{messages_text}"
            save_to_supabase = False
            summary_tracker = await self.create_summary(
                content, save_to_supabase
            )
            formatted_summary = self.format_summary(summary_tracker)

            print(f"Formatted summary: {formatted_summary}")
            # Store summary results in AppConfig if we got a valid summary
            if formatted_summary:
                # Store both human readable format and SummaryTracker object
                AppConfig().get_call_metadata()[
                    "latest_message_summary_readable"
                ] = formatted_summary
                # Store json string of SummaryTrackerClass
                AppConfig().get_call_metadata()[
                    "latest_message_summary_tracker"
                ] = json.dumps(summary_tracker.__dict__)
        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            logger.error(
                f"Error in summarize_transcript: {str(e)}\nTraceback: {error_details}"
            )
            print(f"Error in summarize_transcript: {str(e)}")

    def inject_summary(self, formatted_messages: list, agent_name: str) -> list:
        """
        Inject the latest conversation summary into the second message of the conversation.

        Args:
            formatted_messages (list): List of conversation messages

        Returns:
            list: Messages with summary injected into second message
        """
        # Get the latest summary from AppConfig
        summary_msg = (
            AppConfig()
            .get_call_metadata()
            .get("latest_message_summary_readable", "")
        )
        if len(formatted_messages) < 2:
            return formatted_messages
        existing_content = formatted_messages[1].content

        # Check if there's already a summary in the content
        if (
            "Summary of conversation so far:" not in existing_content
            and agent_name
            not in [
                "make_payment",
                "make_payment_with_method_on_file",
                "collect_debit_card",
                "collect_bank_account",
            ]
        ):
            # Add the summary at the beginning if it doesn't exist
            formatted_messages[1].content = f"{summary_msg}{existing_content}"
        return formatted_messages

    def default_summary_prompt(self):
        """Initialize the system prompt for the summary LLM."""
        return """
            You are a precise information extraction system. Your task is to analyze conversations and extract ONLY explicitly confirmed information from human messages.
            ##Core Instructions:

            Extract only explicitly confirmed information from the human (not from AI responses)
            Maintain persistence: If "Summary of conversation so far..." exists, preserve this information unless explicitly modified by new human messages
            Handle agreement carefully: When human gives simple agreement (e.g., "yes", "sure", "okay"), only extract information specifically referenced in the LATEST question they're agreeing to. Do not include amounts or dates that were in the earlier intro.
            Use latest information: When multiple instances of the same information appear, prioritize the most recent confirmed version
            IMPORTANT: Never assume the amount or date. Omit uncertain fields: Leave any field empty if no explicit confirmation exists

            ## Field Definitions:
            desired_payment_amount: float
                - Extract ONLY when the human explicitly confirms a specific amount
                - Guidelines:
                    - When human agrees to AI's question, extract amount ONLY if it was specifically mentioned in that question
                    - For relative amounts (e.g., "half", "a hundred less"), calculate based on most recent total
                    - If human requests a change without specifying new amount, set to null until provided
                    - If no previous summary exists, only update when human explicitly states a new amount
                    - NEVER use AI-suggested amounts that haven't been explicitly confirmed
                    - If human doesn't mention changing an existing confirmed amount, preserve existing value
                    - When calculating relative amounts, use the most recent confirmed amount as the base

            desired_payment_date: str
                - Extract ONLY when the human explicitly confirms a specific date
                - Guidelines:
                    - Preserve exact date formatting as stated by human (e.g., "today", "tomorrow", "next Friday")
                    - When human agrees to AI's question, extract date ONLY if it was specifically mentioned in that question
                    - If human requests a change without specifying new date, set to null until provided
                    - If no previous summary exists, only update when human explicitly states a new date
                    - NEVER use AI-suggested dates that haven't been explicitly confirmed
                    - If human doesn't mention changing an existing confirmed date, preserve existing value
                    - Use the exact date string stated by the human (i.e today, tomorrow) or in the a.i message (do NOT try formatting)

            ## Examples:

            Example 1:
            AI: "The due date was March 5th. Can you pay $500?"
            Human: "yes"
            ✓ Update payment amount to $500
            ✗ DO NOT update the date
            
            Example 2:
            Human: "I can pay half now"
            Previous amount mentioned: $800
            ✓ Update payment amount to $400
            ✗ DO NOT update the date
            
            Example 3:
            Human: "Actually, I'd like to pay a different amount"
            ✓ Set payment amount to null
            ✗ DO NOT update the date
            
            Example 4:
            Human: "I can pay on Friday"
            ✓ Update payment date to "Friday"
            ✗ DO NOT update the amount
            
            Example 5:
            AI: "Would you be able to make a payment today?"
            Human: "sure I can do today"
            AI: "Great! Would you like to pay the total due amount of $297.82 today?"
            Human: "actually let me do tomorrow"
            ✓ Set payment amount to null 
            ✓ Update payment date to "tomorrow"
            Reasoning:
            - The human never agreed to the amount of $297.82, so we set it to null
            - The human confirmed the date of "tomorrow" so we update the date
            ✗ DO NOT pull the ai offered amount of $297.82
            
            Example 6:
            AI: "Currently, your account is 5 days past due for $304.96. Would you be able to make a payment today?"
            Human: "yes"
            ✓ Set payment amount to null
            ✓ Update payment date to "today"
            Reasoning:
            - The human only confirmed the date of "today" but did not confirm the amount of $304.96
            - The amount was only mentioned as context in the AI's message, not as part of the question being asked
            ✗ DO NOT pull the ai mentioned amount of $304.96
            
            Example 7:
            AI: "Currently, your account is 8 days past due for $285.19. Would you be able to make a payment today?"
            Human: "I can do 250"
            AI: "Alright, and when would you like to make this payment of $250?"
            Human: "wait no I can only pay half that sorry"
            ✓ Update payment amount to $125.0
            ✓ Set payment date to null
            Reasoning:
            - The human first confirmed $250 as the amount
            - Then they changed their mind and said they can only pay "half that"
            - "half that" refers to the most recent confirmed amount ($250), not the original amount ($285.19)
            - The human never confirmed a date, so we set it to null
            ✗ DO NOT use the original amount ($285.19) as the base for calculating half

            Example 8:
            tool: Collect Bank Account Assistant. You are currently collecting bank account information for the payment of ($400.0) on (today).
            AI: "Thank you, Luz, for agreeing to make a payment today. Can you plese provide your bank account number?"
            Human: "actually can i do two hundred dollars instead"
            ✓ Update payment amount to $200.0
            ✓ Keep payment date as "today"
            Reasoning:
            - The human changed the amount from 400 to $200
            - The human did not mention changing the date, so we preserve the previously confirmed date of "today"
            - The bank discussion does not affect the amount or date
            ✗ DO NOT set the date to null just because the payment method was discussed
            
            Example 9:
            AI: "Currently, your account is 15 days past due for $336.51. Would you be able to make a payment today?"
            Human: "what are you talking about?"
            AI: "I'm here to assist you with your Exeter Finance auto loan. Your account is currently 15 days past due with a total due amount of $336.51. Would you like to make a payment today?"
            Human: "hmm, i cant pay that much"
            AI: "I understand. Would you like to make a partial payment of $168.25 today?"
            Human: "oh wow, i can pay that amount"
            ✓ Update payment amount to $168.25
            ✓ Update payment date to "today"
            Reasoning:
            - The human agreed to the specific amount of $168.25 and date of "today" in the AI's question
            - The human's response "oh wow, i can pay that amount" confirms both the amount and date since they were both in the question
            - The earlier context about $336.51 is irrelevant since the human explicitly rejected that amount
            ✗ DO NOT use the original amount of $336.51
            
            Example 10:
            AI: "Currently, your account is 12 days past due for $342.86. Would you be able to make a payment today?"
            Human: "how about in a week"
            ✓ Set payment amount to null
            ✓ Update payment date to "in a week"
            Reasoning:
            - The human provided a specific date "in a week" but did not mention any amount
            - The amount of $342.86 was only mentioned as context in the AI's message, not as part of the question being asked
            - Since the human did not confirm any amount, we set it to null
            ✗ DO NOT pull the ai mentioned amount of $342.86
        """

    def ally_summary_prompt(self):
        """Initialize the system prompt for the summary LLM."""
        return """
            You are a precise information extraction system. Your task is to analyze conversations and extract ONLY explicitly confirmed information from human messages, specifically for Ally's payment scenarios which may involve multiple payments.

            ## Core Instructions:

            Extract only explicitly confirmed information from the human (not from AI responses)
            Maintain persistence: If "Summary of conversation so far..." exists, preserve this information unless explicitly modified by new human messages
            Handle agreement carefully: When human gives simple agreement (e.g., "yes", "sure", "okay"), only extract information specifically referenced in the LATEST question they're agreeing to
            Use latest information: When multiple instances of the same information appear, prioritize the most recent confirmed version
            IMPORTANT: Never assume amounts or dates. Omit uncertain fields: Leave any field empty if no explicit confirmation exists
            Track multiple payments: Be prepared to handle scenarios where the customer may split their payment into two separate payments
            IMPORTANT: If payment amount 1 is being changed, set payment amount 2 to null while keeping all dates unchanged
            IMPORTANT: When customer wants to change an amount/date but doesn't specify the new value, set the corresponding field to null until they provide the new value
            IMPORTANT: If the AI's question includes a specific date and the human responds with an amount, consider the date confirmed as well
            IMPORTANT: Changing dates does NOT affect payment amounts, and changing payment amount 1 only clears payment amount 2, not dates
            IMPORTANT: Only add change notes when modifying an existing summary, not for new payment plans
            IMPORTANT: When customer mentions "full amount" or similar phrases, calculate it as the sum of all existing payment amounts

            ## Field Definitions:

            desired_payment_amount_1: float
                - Extract ONLY when the human explicitly confirms a specific amount for the first payment
                - Guidelines:
                    - When human agrees to AI's question, extract amount ONLY if it was specifically mentioned in that question
                    - For relative amounts (e.g., "half", "a hundred less"), calculate based on most recent total
                    - If human requests a change without specifying new amount, set to null until provided
                    - If this amount is changed, payment amount 2 must be set to null (but keep all dates unchanged)
                    - If human wants to change the amount but doesn't specify the new value, set to null
                    - If human mentions "full amount", calculate it as the sum of all existing payment amounts

            desired_payment_date_1: str
                - Extract ONLY when the human explicitly confirms a specific date for the first payment
                - Guidelines:
                    - Preserve exact date formatting as stated by human (e.g., "today", "tomorrow", "next Friday")
                    - When human agrees to AI's question, extract date ONLY if it was specifically mentioned in that question
                    - If human requests a change without specifying new date, set to null until provided
                    - If the AI's question includes a specific date and the human responds with an amount, consider the date confirmed as well
                    - If human wants to change the date but doesn't specify the new value, set to null
                    - Changing this date does NOT affect any payment amounts

            desired_payment_amount_2: float
                - Extract ONLY when the human explicitly confirms a specific amount for the second payment
                - Follow same guidelines as desired_payment_amount_1
                - If only one payment is discussed, leave this field empty
                - If payment amount 1 is changed, this field must be set to null (but keep all dates unchanged)
                - If human wants to change the amount but doesn't specify the new value, set to null

            desired_payment_date_2: str
                - Extract ONLY when the human explicitly confirms a specific date for the second payment
                - Follow same guidelines as desired_payment_date_1
                - If only one payment is discussed, leave this field empty
                - If human wants to change the date but doesn't specify the new value, set to null
                - Changing this date does NOT affect any payment amounts

            customer_desired_change_notes: str
                - Track any requests from the customer to change previously confirmed payment dates or amounts
                - Only populate when customer explicitly requests a change to an existing summary
                - Do NOT populate for new payment plans
                - Clear this field after each summarization to avoid persisting old change requests
                - When specific new value is provided, format as: "Customer wants to change first payment amount from $300 to $200"
                - When no specific value is provided, format as: "Customer wants to change first/second payment amount/date, no specified value"
                - For multiple unspecified changes, format as: "Customer wants to change first payment amount and date, no specified values"
                - For multiple specified changes, format as: "Customer wants to change first payment amount from $200.0 to $357.04 and date from wednesday to today"

            ## Examples:

            Example 1:
            AI: "Can you pay $500 today?"
            Human: "yes"
            ✓ Update payment amount 1 to $500 and date 1 to "today"
            ✗ DO NOT populate payment 2 fields
            ✗ DO NOT populate change notes since this is a new payment plan

            Example 2:
            Human: "I can pay $250 on March 5th and $250 on March 10th"
            ✓ Update payment amount 1 to $250 and date 1 to "March 5th"
            ✓ Update payment amount 2 to $250 and date 2 to "March 10th"
            ✗ DO NOT populate change notes since this is a new payment plan

            Example 3:
            Previous summary shows: Payment 1: $300 on March 5th, Payment 2: $300 on March 10th
            Human: "I need to change the amount"
            ✓ Set payment amount 1 to null
            ✓ Set payment amount 2 to null
            ✓ Keep both dates unchanged
            ✓ Add to change notes: "Customer wants to change first payment amount, no specified value"

            Example 4:
            AI: "I see you have a total amount due of $1218.32. Can you pay that today?"
            Human: "oh thats a lot"
            AI: "I understand it can feel like a lot. What amount would you be able to pay today?"
            Human: "i can do 700 dollars"
            ✓ Update payment amount 1 to 700.0 and date 1 to "today"
            ✗ DO NOT populate payment 2 fields
            ✗ DO NOT populate change notes since this is a new payment plan
            Reasoning:
            - The AI's question included "today" and asked for an amount
            - The human responded with an amount, implicitly confirming the date "today"

            Example 5:
            Previous summary shows: Payment 1: $600 on April 16th, Payment 2: $298.67 on 20th
            Human: "I want to make the first payment on a different date"
            AI: "Sure, I can help with that. What new date would you like to set for your first payment of $600?"
            Human: "today"
            ✓ Keep payment amount 1 as $600
            ✓ Update payment date 1 to "today"
            ✓ Keep payment amount 2 as $298.67
            ✓ Keep payment date 2 as "20th"
            ✓ Add to change notes: "Customer wants to change first payment date from April 16th to today"
            Reasoning:
            - Only the date is being changed, so payment amounts remain unchanged
            - Payment amount 2 should not be cleared since only a date was changed

            Example 6:
            AI: "I see you have a total amount due of $1242.09. Can you pay that today?"
            Human: "I can pay 1000"
            AI: "Thank you for that. Since the amount is $1000, there will still be a remaining balance of $242.09. When would you be able to make this second payment?"
            Human: "I want first payment on monday"
            ✓ Update payment amount 1 to 1000.0
            ✓ Update payment date 1 to "monday"
            ✗ DO NOT populate payment 2 fields
            ✗ DO NOT populate change notes since this is a new payment plan
            Reasoning:
            - This is a new payment plan being established
            - No previous summary exists to modify
            - Therefore, no change notes should be added

            Example 7:
            Previous summary shows: Payment 1: $200.0 on wednesday, Payment 2: $157.04 on thursday
            Human: "hm actually ill pay full amount today"
            ✓ Update payment amount 1 to 357.04 (sum of $200.0 + $157.04)
            ✓ Update payment date 1 to "today"
            ✓ Set payment amount 2 to null
            ✓ Set payment date 2 to null
            ✓ Add to change notes: "Customer wants to change first payment amount from $200.0 to $357.04 and date from wednesday to today"
            Reasoning:
            - "full amount" means the sum of all existing payments ($200.0 + $157.04 = $357.04)
            - "today" is explicitly stated as the new date
            - Since payment amount 1 is being changed, payment amount 2 must be set to null
            - This is a change to an existing summary, so change notes should be added
        """
