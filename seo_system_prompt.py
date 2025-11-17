import os
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Literal

from dotenv import load_dotenv

load_dotenv()

# --- Pydantic Model for Structured Output ---
# This class defines the exact structure we want the LLM to return.
# The user's original class definition is used here.
class State(BaseModel):
    """
    Classifies the user's query into 'seo' or 'other'.
    A query can belong to one or both categories.
    """
    query_type: List[Literal['seo', 'other']] = Field(
        ...,
        description="A list containing 'seo', 'other', or both, based on the user's query."
    )

# --- System Prompt ---
# This is the system prompt you requested. It instructs the LLM on its
# specific task, focusing on classification and the required output format.
SYSTEM_PROMPT = """
Your sole function is to analyze the user's query and classify its intent.
You must determine if the query is related to:
1. 'seo': Any topic explicitly or implicitly connected to SEO.
2. 'other': Any topic not related to SEO.

A query can fall under one or both categories.
- If a query is ambiguous and could plausibly relate to SEO, include both 'seo' and 'other'.
- If a query clearly does not relate to SEO, include only 'other'.
- If a query clearly relates to SEO, include only 'seo'.

You MUST respond only with the required JSON structure.
Do not add any conversational text or explanations.
"""

def classify_query(user_query: str) -> State:
    """
    Calls the OpenAI API to classify a user query using structured output.
    """
    # Ensure the OPENAI_API_KEY environment variable is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it by running: export OPENAI_API_KEY='your_api_key_here'")
        return State(query_type=[]) # Return empty state on error

    # Initialize the OpenAI client
    # It automatically reads the OPENAI_API_KEY from environment variables
    client = OpenAI()

    print(f"Attempting to classify query: \"{user_query}\"")

    try:
        # This is the key call for structured output.
        # We pass our Pydantic `State` class to the `response_model` parameter.
        completion = client.chat.completions.parse(
            model="gpt-4o",  # You can also use "gpt-4-turbo" or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ],
            # This tells the OpenAI client to expect a JSON response
            # that conforms to the `State` Pydantic model.
            response_format=State,
        )
        
        # The 'completion' variable is now an instance of your Pydantic 'State' class
        return completion.choices[0].message.parsed

    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return State(query_type=[]) # Return empty state on error

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Purely SEO
    query1 = "How do I improve my website's Google ranking and find good keywords?"
    classification1 = classify_query(query1)
    print(f"Classification for query 1: {classification1.model_dump_json(indent=2)}\n")

    # Example 2: Purely Other
    query2 = "What's the weather like in London today?"
    classification2 = classify_query(query2)
    print(f"Classification for query 2: {classification2.model_dump_json(indent=2)}\n")

    # Example 3: Both SEO and Other
    query3 = "Can you explain what a SERP is and also tell me the capital of France?"
    classification3 = classify_query(query3)
    print(f"Classification for query 3: {classification3.model_dump_json(indent=2)}\n")

    # Example 4: Ambiguous/Other
    query4 = "Help me with my blog."
    classification4 = classify_query(query4)
    print(f"Classification for query 4: {classification4.model_dump_json(indent=2)}")