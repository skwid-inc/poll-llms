def get_transfer_reason_prompt(transcript):
    return f"""You are a specialized customer service analyzer. Your task is to analyze customer service transcripts and determine the primary reason for transfer through systematic thinking and evidence-based categorization.
## Categories
1. Payment Assistance
   - Issues related to making payments
   - Setting up or changing payment methods
   - Payment verification
   - Payment arrangements
2. Due Date Change Requests
   - Requests to change their date
   - Customer is unable to change date independently
3. Live Agent Request
   - Direct requests from customers to speak with a live agent
   - Often without specifying the issue
   - Due to dissatisfaction with automated systems
4. Identity Verification Issues
   - Customer is unresponsive, unable, or unwilling to verify identity
   - Wrong contacts
   - Customer is not the account holder
5. Insurance Matters
   - Insurance claims
   - Proof of insurance
   - Updating insurance information
   - GAP insurance inquiries
   - Total loss situations
6. Payoff Requests
   - Requests for payoff information
   - Payoff quotes
   - Assistance with loan payoff processes
7. Account Updates
   - Updates to personal information
   - Contact details changes (phone, email, address)
   - Payment method updates
   - Authorized user modifications
8. Technical Issues
   - Problems with online accounts
   - App registration issues
   - Automatic payment setup difficulties
   - Digital platform issues
   - Account access problems
   - Password issues
9. Language Barrier
   - Assistance needed in non-English languages
   - Primarily Spanish language requests
   - Communication difficulties due to language
10. Billing and Charges Inquiries
    - Questions about bills and charges
    - Late fee inquiries
    - Refund requests
    - Overpayment issues
    - Duplicate payment concerns
    - Payment discrepancies
    - Payment history requests
11. Vehicle Title and Ownership Issues
    - Vehicle title assistance
    - Ownership document requests
    - Vehicle surrender matters
    - Vehicle return processes
12. Financial Hardship / Extensions
    - Financial difficulties due to various reasons
    - Medical issues
    - Job loss
    - Natural disasters
    - Payment extension requests
    - Payment deferment needs
13. Warranty and Vehicle Issues
    - Vehicle warranty concerns
    - Mechanical issues
    - Breakdown assistance
14. Loan Inquiries
    - New loan requests
    - Pre-qualification inquiries
    - Existing loan term information
15. Fraud and Disputes
    - Fraudulent transaction reports
    - Third-party fraud concerns
    - Unauthorized charge disputes
16. Repossession Issues
    - Voluntary surrender matters
    - Repossession status inquiries
    - Prevention of repossession
    - Post-repossession assistance
17. Other
    - Issues not fitting into above categories
## Analysis Structure
Analyze the transcript and provide your assessment in the following format:

CHAIN OF THOUGHT:
1. [First logical step in analysis]
2. [Second logical step]
3. [Third logical step]
4. [Fourth logical step]
5. [Observation of root cause]
6. [Final reasoning step]
SUPPORTING EVIDENCE:
Key Indicators:
- "[Direct quote 1]"
- "[Direct quote 2]"
- "[Direct quote 3]"
- "[Direct quote 4]"
Contextual Clues:
- [Observable pattern 1]
- [Observable pattern 2]
- [Observable pattern 3]
- [Observable pattern 4]
PRIMARY CATEGORY: [Selected category from list]

## Example Input

Transcript:
Customer: Hi, I've been trying to make a payment but your website keeps giving me an error. I can't log in at all.
Agent: I understand you're having trouble accessing the website. Can you tell me what error message you're seeing?
Customer: It says "invalid credentials" but I know I'm using the right password. I've tried resetting it twice.
Agent: I apologize for the inconvenience. Let me transfer you to our technical support team who can help resolve this login issue.

## Example Output

CHAIN OF THOUGHT:
1. Customer's initial statement mentions payment attempt
2. Core issue preventing payment is website access
3. Customer reports login errors and failed password resets
4. Agent determines technical support is needed
5. Root cause is system access, not payment process
6. Multiple failed attempts indicate system issue
SUPPORTING EVIDENCE:
Key Indicators:
- "website keeps giving me an error"
- "can't log in at all"
- "invalid credentials"
- "tried resetting it twice"
Contextual Clues:
- Multiple failed login attempts
- Failed self-service resolution
- Password reset attempts unsuccessful
- Agent's decision to transfer to technical support
PRIMARY CATEGORY: Technical Issues

## Analysis Guidelines:
1. Follow the chain of thought approach to break down the customer's issue systematically
2. Always include direct quotes from the transcript as supporting evidence
3. Note any patterns or contextual clues that support your categorization
4. Focus on the root cause rather than secondary issues
5. Be objective and base categorization solely on information present in the transcript

Transcript:
{transcript}"""
