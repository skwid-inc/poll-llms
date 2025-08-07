general_taylor_context = """Here's some context on the conversations you will be analyzing (this is also in the agent's <SystemPrompt>). You have incoming or outgoing calls from Taylor, an AI assistant representing various financial institutions. In outgoing calls Taylor calls a number associated with a specific given name, has details of customer's account such as due amounts, dates, fees and various instructions on how to deal with the specific customer during the call. All info related to a customer is mentioned under "Context:". The agent also calls various tools and agents to pull up, validate, process information and service the loan. The agent calls the number asks to identify/confirm the customer then proceeds with the call purpose. Incoming calls are from a customer to the agent and proceed differently."""


CUSTOMER_CALLBACK_JUDGE = {
    "system_prompt": """You are a judge that detects from an agent's <Agent Current Response>, if the agent unambiguously mentions that a live agent will call back (customer will be contacted separately in a subsequent call). Do not detect if the agent is indicating it will transfer the current call to a live agent.
{general_taylor_context}

Return JSON: {{"callback_mentioned": Boolean}}
""".format(general_taylor_context=general_taylor_context),
    "instructions": """
**Detect responses that indicate callbacks from a live agent, which are equivalent in meaning to the below:**
- "A representative will call you back", "You will be contacted by a live agent", "A human will reach out to you"

**Do NOT detect if its not actually about a callback from a live agent:**
- "will transfer you to a live agent", "let me transfer you" (these indicate that this current call is being transferred to a live agent)
- "will be in touch with you shortly" (this one is ambiguous and can indicate both callbacks and transfers so we don't detect)
- messages unrelated to callbacks from a live agent""",
}


PAYMENT_PROCESSING_LLM_JUDGE_PROMPT = {
    "system_prompt": """You are an expert judge tasked with precisely detecting if an agent's message involves it *claiming*, *for the FIRST time*, that an individual payment, from the user within that conversation, has been setup within this conversation, given conversation context.
{general_taylor_context}

You provide your analysis in the following JSON format:
{{
    "explanation": "string explaining your reasoning for the payment status",
    "payment_confirmed": Boolean
}}

You carefully reason over and evaluate each of the rules of your task. If the input is not in English, reason with the same rules, and give final answer and explanation in English.
""".format(general_taylor_context=general_taylor_context),
    "instructions": """Analyze the following single conversation turn under these rules.

<INSTRUCTIONS>
## Primary objective
Detect if an agent is claiming, for the FIRST time, that an individual payment from the user has been setup, scheduled, or is being processed.

## What to detect
Look for messages where the agent confirms a payment has been setup, scheduled, is being processed or confirms a user's promise to pay. Focus on detecting only the current turn, not the full conversation (which is only for context).

## Be careful about:
1. **First Time Only**: Only detect if this is the FIRST time this payment is being confirmed. If the payment was already clearly confirmed in prior conversation context, do not detect again.
2. **Current Response Only**: Detect only in the current target response. Use conversation context only to determine if this payment was already confirmed previously.
3. **Explicit Confirmation Required**: The agent must explicitly confirm the payment is setup/scheduled/processed. Ambiguous statements should not be detected.
4. This must be a payment that is made *within this conversation*: so, if the discussion is about a payment made elsewhere, or corresponds to a payment mentioned in SystemPrompt, do not detect.

## What NOT to detect
- **Autopay**: Ignore ANYTHING related to automatic payment setup
- **Processing steps**: If the agent talks about performing steps to set up the payment, do not detect (indicates ambiguity)
- **System prompt information**: If the agent confirms facts about prior payments already mentioned in the SystemPrompt context, do not detect. Read the dates and numbers in system prompt carefully to check for matches.
- **Ambiguous confirmations**: If the agent says payment is scheduled but also asks for payment details in the same message, do not detect

## Step-by-step detection process
Apply these rules in order:
1. **Check if payment-related**: If the message is not relevant to payments being made, return False
2. **Check history**: 
    - Check prior payment confirmations from the agent: If this payment was already clearly confirmed in prior context, do not detect
    - Check <SystemPrompt> context: If the system prompt already contains this information, do not detect. Check the Context: section carefully, which has details like amounts, fees, last payment details etc.
    - Check if autopay: if prior context indicates this is related to autopay, do not detect
3. **Check for explicit confirmation**: Ensure the agent explicitly confirms the payment is setup/scheduled/processed as a necessary condition to detect it
4. **Check for ambiguity**: If the statement indicates the payment is still being setup, by indicating future actions to be taken in the conversation, do not detect. e.g. if the agent mentions scheduling the payment but also asks for payment details, do not detect

## Context usage
Use conversation context only to determine if this payment was already confirmed previously. Do not use context to detect confirmations in previous messages—only analyze the current target response.

## Final instructions
Apply each step systematically in the exact order provided. When analyzing the target response, focus on whether the agent is making a first-time explicit confirmation of payment setup. Remember that some target responses for detection with correct labels may be provided as examples—use their labels and explanations if relevant to your analysis.

Some target <Agent Current Response> responses for detection with the correct labels are given next. For the given response you have to detect, carefully check each one if any of the examples below are relevant, and if so, use their labels and explanation.

</INSTRUCTIONS>

Examples of what IS a payment confirmation (potentially detected):
    - Explicit confirmation that payment is already scheduled, setup, or being processed
    - Clear statements that payment will be processed using specific methods
    - Thank you messages that imply payment was completed (when not previously confirmed)
Examples of what is NOT a payment confirmation (never detected):
    - Statements that indicate things are still being setup (Let me take care of that for you)
        - e.g. statements that mention payment is scheduled scheduling but also ask customer for details, indicating the payment is still being setup (How would you like to make this payment? Would you like to use your debit card ending in 1234 to make this payment?)
    - Automatic payment setup confirmations
    - Processing steps without explicit confirmation
    - Restating information from system prompt context

Examples with payment_confirmed: True and explanation labels:

<short_examples>
- "Your payment for the full amount of $375.48 is scheduled to be processed on June 27th." -> (True, "Explicit confirmation: payment is scheduled for processing")
- "Your payment of $278.53 is being processed." -> (True, "Explicit confirmation: payment is currently being processed")
- "Your payment of $278.53 is being processed. If there's anything else you need, feel free to ask!" -> (True, "Explicit confirmation: payment is being processed, can infer from context that nothing else is requested")
- "Your payment has successfully been set up for tomorrow, June 1st, 2025, and your confirmation number is 3456788." -> (True, "Explicit confirmation: payment is scheduled with confirmation number")
- "Your payment will be processed using the method on file." -> (True, "Explicit confirmation: payment is setup for processing")
- "Thank you for your payment." -> (True, "Implied confirmation: thanking for completed payment")
- "Thank you for your payment. If you have any other questions or need further assistance in the future, feel free to reach out. Have a great day!" -> (True, "Implied confirmation: thanking for completed payment when not previously confirmed")
- "Your payment will be processed using the bank account ending in 7117." -> (True, "Explicit confirmation: payment is setup for processing")
- "I understand. I've noted that you plan to call back on Friday to make the payment manually. Is there anything else I can assist you with today?" -> (True, "Notate to pay: noting promise, confirming setup")
</short_examples>

Examples with payment_confirmed: False and explanation labels:

<short_examples>
- "Your request to schedule a payment of $50.10 for June 1st has been successfully processed." -> (False, "Request processing confirmed, not payment itself")
- "Great. Let me process that payment for you." -> (False, "Processing step mentioned, not explicit confirmation")
- "Thank you for letting me know. I can help you set up a promise to pay arrangement for this coming Friday, the twenty-fifth. Let me take care of that for you." -> (False, not already confirmed")
- "Your automatic payment setup is complete." -> (False, "Autopay: automatic payment setup, not individual payment")
- "Your autopayment is setup is successfully processed for the 4th of April, 2025." -> (False, "Autopay: automatic payment setup, not individual payment")
- "Great. Give me a second to pull my debit card capture form." -> (False, "Processing step: no explicit confirmation")
- "I'll proceed with scheduling your payment of $100 using the bank account ending in 5678." -> (False, "Processing step: scheduling mentioned but not confirmed")
- "Thank you. I will schedule a payment of $528.01 for January 27th. Checking my system." -> (False, "Processing step: scheduling mentioned but not confirmed")
- "Yes, the payment is scheduled for Friday. Would you like to proceed with using the VISA ending in 3999?" -> (False, "Ambiguous: mentions scheduling but asks for payment details")
</short_examples>

<example>
<Conversation Context>
<USER>: already paid off my payment but it says it's three days past due
</Conversation Context>
<Agent Current Response>I understand that you have already paid off your payment. Please wait for it to go through, then.</Agent Current Response>

{payment_confirmed: False, explanation: "Agent is acknowledging user's message, not making a new confirmation"}
</example>

<example>
<Conversation Context>
<USER>: set up payment already
</Conversation Context>
<Agent Current Response>I understand. Since the payment is set up, if you need any further assistance or changes, feel free to reach out.</Agent Current Response>

{payment_confirmed: False, explanation: "Agent is acknowledging user's statement, not making a new confirmation"}
</example>

<example>
<Conversation Context>
<SystemPrompt>... Westlake Financial has received a total loss payment of $3725.77 from the insurance company. ...</SystemPrompt>
</Conversation Context>
<Agent Current Response>Regarding your insurance claim, Westlake Financial has received a total loss payment of $3725.77 from the insurance company.</Agent Current Response>

{payment_confirmed: False, explanation: "Agent restates information from system prompt context, not confirming user payment"}
</example>

<example>
<Conversation Context>
<SystemPrompt>... Context: Customer's delinquent amount is $0.00, monthly payment is $264.36. Late fees: $0.0, GAP fees: $None, NSF fees: $0.0. Due date: the second of August, 2025, 0 days past due. Last payment: $264.36 on 2025-07-02. Half of delinquent amount: $0.00. ...</SystemPrompt>
</Conversation Context>
<Agent Current Response>Your payment of $264.36 was successfully processed on July 2, 2025</Agent Current Response>

{payment_confirmed: False, explanation: "Agent restates information from system prompt context, not confirming new payment"}
</example>

<example>
<Conversation Context>
<AGENT>: Your payment has successfully been set up for 5th June 2025.
</Conversation Context>
<Agent Current Response>Thank you for your payment! If you have any more questions or need assistance in the future, feel free to reach out. Have a great day!</Agent Current Response>

{payment_confirmed: False, explanation: "Payment was already confirmed previously, this is not a first-time confirmation"}
</example>

<example>
<Conversation Context>
<AGENT>: Your payment is being processed using the debit card ending in 1234.
<AGENT>: Are you on the phone with me?
<USER>: yes
</Conversation Context>
<Agent Current Response>Thank you for your payment!</Agent Current Response>

{payment_confirmed: False, explanation: "Payment was already confirmed to be processing previously, this is a repeated confirmation"}
</example>
""",
}


MATHEMATICAL_CALCULATION_LLM_JUDGE_PROMPT = {
    "system_prompt": """You are an expert judge tasked with detecting if an agent's message involves performing clear mathematical calculations based on the conversation context.
{general_taylor_context}

You provide your analysis in the following JSON format:
{{
    "explanation": "string explaining your reasoning for whether mathematical calculation is detected",
    "math_detected": Boolean
}}

You carefully reason over and evaluate each of the rules of your task. If the input is not in English, reason with the same rules, and give final answer and explanation in English.
""".format(general_taylor_context=general_taylor_context),
    "instructions": """Analyze the following single conversation turn under these rules.

<INSTRUCTIONS>
## Primary objective
Determine if an agent is performing mathematical computations, implicitly or explicitly, in their current response and detect if it passes the checks mentioned below.
- Avoid detecting extremely trivial calculations. 
- For something to be considered a calculation and detected, the result has to be *explicitly stated*. 
- Date-day conversions are considered calculations.

Steps:
1. Check the current message message wrt prior context for calculations involving arithmetic operations, date arithmetic, percentage calculations, payment amount computations, fee calculations, discount calculations, etc.
    - if these aren't the case, do not detect.
    - if the calculation is extremely trivial, do not detect.
    - if the result is not explicitly stated, do not detect.
2. Compare with specific prior context:
    - check the prior AGENT, USER, SystemPrompt and TOOL outputs for earlier mentions.
    - if the agent was simply confirming a number or date that was already mentioned in prior context, do not detect, as this is not a calculation.
    - if the agent repeats anything that was already confirmed in SystemPrompt or TOOL outputs, NEVER detect these. These are considered trustworthy and the agent may freely assume/confirm these.
3. Check each of the given examples below for relevant ones to use their labels and explanations. Keep in mind all the <BE_CAREFUL> rules.
4. Once you have checked all these, make your decision.


<BE_CAREFUL>
- The result of a calculation must be explicitly stated for it to be detected.
- Detect only in the current target response. Only check the conversation context to help determine if in the current turn, the agent is deriving values through computation or confirming user calculations. Do not use context to detect calculations in previous messages—only analyze the current target response.
- To detect if something is a calculation, you don't care about if the result is correct.
- Check <SystemPrompt>/<TOOL> outputs: If the calculation/conversion/numbers come from the TOOL or SystemPrompt outputs, it should not be detected even if agent message looks like a calculation. These SystemPrompt and TOOL outputs are considered trustworthy and the agent may freely assume/confirm these. DO NOT DETECT in these cases.
- If the agent is just re-stating a *number* made from prior SystemPrompt, TOOL, USER or AGENT messages, it mustn't be detected.
- If the agent confirming a *calculation* from the prior USER or AGENT messages, you should consider detecting it if the other checks have passed. 
- Stating a date with the day could be an implicit calculation unless verified in prior context. E.g. If the user says only "Friday" and the agent says "Friday the 30th", this is an implicit calculation that assumes that 30th is Friday. It might have to be detected.


Some target <Agent Current Response> responses for detection with the correct labels are given next. For the given response you have to detect, carefully check each one if any of the examples below are relevant, and if so, use their labels and explanation. 

</INSTRUCTIONS>

Examples of what IS a mathematical calculation (potentially detected):
    - Computing total amount by adding fees to base amount, adding up amounts of multiple payments made, remaining balance by subtracting payment from total, etc.
    - Computing payment schedules or installment amounts
    - Computing percentages or applying discounts
    - date arithmetic (e.g., "July 15th is 30 days from today")
    - Adding or subtracting days from dates
    - statements that report an explicit result of the calculation even though the computation itself may be implicit.
Examples of what is NOT a mathematical calculation (never detected):
    - Simply stating a known amount or date or other details given in context
    - statements that don't report an explicit result of the calculation
    - Extremely trivial things:
        - Converting date/number formats without arithmetic (e.g., "June 15th" to "06/15", one thousand to 1000)
        - Simple inference from given information: This is July. The 16th of this month is July 16th.
    - Stating policy information without calculations: e.g. "there is a 5% late fee"

Examples with math_detected: True and explanation labels:

<short_examples>
- "Your total amount due is $500 plus a $25 late fee, which comes to $525." -> (True, "Arithmetic calculation: adding late fee to base amount")
- "Your payment date is June 15th, 2024." -> (False, "Simply stating a known date without any computation")
- "If you pay $200 today, your remaining balance will be $300." -> (True, "Arithmetic calculation: subtracting payment from total balance")
- "Your payment of $150 plus the 3% processing fee of $4.50 totals $154.50." -> (True, "Multi-step calculation: percentage computation plus addition")
- "I understand you're looking to schedule a payment for in 2 days and while keeping 100$ left in balance." -> (False, "Mentioning timeframe without computing specific result")
- "The processing fee is $25." -> (False, "Simply stating a known fee amount")
- "Your 2% discount on the $500 balance would be $10, making your total $490." -> (True, "Multi-step calculation: percentage discount then subtraction")
- "The grace period is 10 days, so your payment is due by June 25th." -> (True, "Date arithmetic: computing due date from grace period")
</short_examples>

Note that in all the above examples where detection is True, if there had been prior context where the numbers were validated through SystemPrompt or TOOL outputs, the detection could have been False.

<example>
<Conversation Context>
<USER>: What's my balance?
</Conversation Context>
<Agent Current Response>Your balance is $500.</Agent Current Response>

{"math_detected": False, "explanation": "Simply restating known value from context without any computation"}
</example>

<example>
<Conversation Context>
<USER>: I want to pay on the tenth
</Conversation Context>
<Agent Current Response>Certainly, we can schedule the payment for Thursday, July 10th, 2025. How much would you like to pay on that date?</Agent Current Response>

{"math_detected": True, "explanation": "Date arithmetic: mapping day name to specific day and date. Stating the day and date assumes they are correctly matched, an implicit calculation."}
</example>

<example>
<Conversation Context>
<USER>: What if I pay $200 today?
</Conversation Context>
<Agent Current Response>Your remaining balance would be $300.</Agent Current Response>

{"math_detected": True, "explanation": "Arithmetic calculation: subtracting payment amount from total balance"}
</example>


<example>
<Conversation Context>
<SystemPrompt> ... Context: Customer's delinquent amount is $375.48, monthly payment is $420.98. Late fees: $2.51, NSF fees: $30.0. ... </SystemPrompt>
<USER>: whats my total?
</Conversation Context>
<Agent Current Response>The total amount due, including your usual monthly payment and any applicable fees, is $453.49. Would you like to make this payment today?</Agent Current Response>

{"math_detected": True, "explanation": "Multi-step calculation: adding up multiple values from context to compute total"}
</example>


<example>
<Conversation Context>
<SystemPrompt> ... Context: Customer's delinquent amount is $200, half-delinquent amount is $100. ... </SystemPrompt>
</Conversation Context>
<Agent Current Response>I can help make payment of the half-delinquent amount, which is $100.</Agent Current Response>

{"math_detected": False, "explanation": "Agent is not performing any calculation, the value is already available in system prompt"}
</example>


<example>
<Conversation Context>
<AGENT>: Your payment of $200 is successfully processed.
<AGENT>: Your payment of $367.34 has been scheduled for the 15th of July, 2025.
</Conversation Context>
<Agent Current Response>I see that you've successfully made two payments today, totaling $567.34. If there's anything else you need assistance with, please let me know.</Agent Current Response>

{"math_detected": True, "explanation": "Arithmetic calculation: adding up multiple payment amounts from context"} (assuming $567.34 was never mentioned in prior context)
</example>


<example>
<Conversation Context>
<USER>: What if I pay in 30 days?
</Conversation Context>
<Agent Current Response>If you pay in 30 days, that would be July 15th.</Agent Current Response>

{"math_detected": True, "explanation": "Date arithmetic: computing specific date from relative timeframe"}
</example>


<example>
<Conversation Context>
<USER>: need another four days to do the pay
</Conversation Context>
<Agent Current Response>Let's setup a payment for four days from now. Can you confirm the amount you'd like to pay?</Agent Current Response>

{"math_detected": False, "explanation": "Mentioning timeframe without computing specific result"}
</example>


<example>
<Conversation Context>
<USER>: thursday may 22nd
</Conversation Context>
<Agent Current Response>Thank you for letting me know. Would you like to schedule a payment for Thursday, May 22nd?</Agent Current Response>

{"math_detected": False, "explanation": "Agent is simply confirming user's stated date, and isn't doing any calculation by itself"}
</example>


<example>
<Conversation Context>
<SystemPrompt>Today is the tenth of July, 2025.</SystemPrompt>
<USER>: I want to pay on the sixteenth
</Conversation Context>
<Agent Current Response>Thank you for letting me know. Would you like to schedule a payment for 16th July?</Agent Current Response>

{"math_detected": False, "explanation": "Agent just infers the month from system prompt context, no calculation performed"}
</example>


<example>
<Conversation Context>
<USER>: What's my total with all fees?
</Conversation Context>
<Agent Current Response>Your balance is $500, plus a 3% processing fee of $15, and a $25 late fee, bringing the total to $540.</Agent Current Response>

{"math_detected": True, "explanation": "Agent performs multi-step calculation: 3% of $500 = $15, then $500 + $15 + $25 = $540"}
</example>


<example>
<Conversation Context>
<SystemPrompt>Context: Customer's delinquent amount is $375.48, monthly payment is $264.36. Late fees: $25.00, processing fee: $4.50. Total amount due: $404.98.</SystemPrompt>
</Conversation Context>
<Agent Current Response>Your total amount due is $404.98, which includes your delinquent amount of $375.48 plus late fees of $25.00 and processing fee of $4.50.</Agent Current Response>

{"math_detected": False, "explanation": "Agent restates pre-calculated information from system prompt, no calculation performed"}
</example>


<example>
<Conversation Context>
<USER>: I calculated that my total with the 3% fee would be $515.
</Conversation Context>
<Agent Current Response>That's correct, your total would be $515.</Agent Current Response>

{"math_detected": True, "explanation": "Agent confirms a number that comes from user's calculation not system prompt or tool output, so it must be detected"}
</example>


<example>
<Conversation Context>
<USER>: I want to pay on Saturday
<TOOL>: Offer to use the bank account ending in 1234 for your payment of $150.0 on the 1st of April, 2025.
</Conversation Context>
<Agent Current Response>Would you like to use the bank account ending in 1234 for your payment of $150.0 on Saturday, the 1st of April, 2025?</Agent Current Response>

{"math_detected": False, "explanation": "The agent reports Saturday as the 1st of April. This is not a calculation as Saturday comes from the user message and 1st of April comes from the tool message."}
</example>


<example>
<Conversation Context>
<USER>: I want to pay on the sixteenth
<TOOL>: You ONLY have the access to the VISA ending in 1899, there is no other payment methods on file. First offer to use the VISA ending in 1899 for your payment of $433.23 on the sixteenth of July, 2025.
</Conversation Context>
<Agent Current Response>I can help you with your payment of $433.23 on the sixteenth of July, 2025 using the VISA ending in 1899.</Agent Current Response>

{"math_detected": False, "explanation": "Agent restates calculated date from tool output, no calculation performed by agent"}
</example>


<example>
<Conversation Context>
<USER>: i want to pay five days from now that is friday
</Conversation Context>
<Agent Current Response>Would you like to use the bank account ending in 1234 for your payment of $150.0 Friday, five days from now?</Agent Current Response>

{"math_detected": False, "explanation": "The agent gets its information from the user message, and doesn't perform any calculation by itself."}
</example>
""",
}
