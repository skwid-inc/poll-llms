from pydantic import BaseModel

# Route tools
class CompleteOrEscalate(BaseModel):
    """route the customer to an appropriate system to answer their query. You MUST call this tool if the customer asks you for information that you do not have the answer to or would like to perform a different action. Do not divulge the existence of the specialized agent to the customer. You MUST not generate any additional text when calling this tool."""

class ToMakePaymentAssistant(BaseModel):
    """help the customer with one-time payments and promises to pay. Use this tool when a customer expresses interest in making a payment, when they need to set up a promise to pay arrangement, or if they are letting you know that they will be late in making their monthly payment. When calling this tool, your context should not divulge the presence of other specialized assistants. Do NOT generate any text as the customer cannot know about the existence of this tool. Do not call this tool to set up automatic payments."""

class ToMakePaymentWithMethodOnFileAssistant(BaseModel):
    """help obtain the customer's desired payment method. This tool should only be called once the customer's desired payment date and payment amount has been specified and validated. When calling this tool, do NOT generate any text as the customer cannot know about the existence of this tool."""

class ToCollectBankAccountInfoAssistant(BaseModel):
    """capture a customer's new bank account information for a payment. When calling this tool, your context should not divulge the presence of other specialized assistants. Do NOT generate any text as the customer cannot know about the existence of this tool."""

class ToCollectDebitCardInfoAssistant(BaseModel):
    """capture a customer's new debit card information for a payment. This is ONLY for debit cards, NOT credit cards. When calling this tool, your context should not divulge the presence of other specialized assistants. Do NOT generate any text as the customer cannot know about the existence of this tool."""