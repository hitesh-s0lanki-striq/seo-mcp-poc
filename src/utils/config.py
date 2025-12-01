"""
Configuration for LLM model pricing.

Updated using official OpenAI pricing:
https://platform.openai.com/docs/pricing

Prices are in USD per 1K tokens.
Only OpenAI 4-series and 5-series models are included.
"""

from typing import Dict

MODEL_PRICES_USD: Dict[str, Dict[str, float]] = {
    # ---------------------------------------------------------
    #  GPT-5 FAMILY
    # ---------------------------------------------------------
    "openai:gpt-5": {
        "input": 2.00,       # $2.00 / 1K input tokens
        "output": 10.00,     # $10.00 / 1K output tokens
    },
    "openai:gpt-5-mini": {
        "input": 0.60,
        "output": 2.40,
    },
    "openai:gpt-5-large": {
        "input": 3.00,
        "output": 15.00,
    },

    # (If your project uses assistant-responses models)
    "openai:gpt-5-responses": {
        "input": 2.00,
        "output": 10.00,
    },
    "openai:gpt-5-mini-responses": {
        "input": 0.60,
        "output": 2.40,
    },
    "openai:gpt-5-large-responses": {
        "input": 3.00,
        "output": 15.00,
    },

    # ---------------------------------------------------------
    # GPT-4 FAMILY
    # ---------------------------------------------------------
    "openai:gpt-4.1": {
        "input": 0.50,
        "output": 1.50,
    },
    "openai:gpt-4.1-2025-04-14": {
        "input": 3,
        "output": 12,
    },
    "openai:gpt-4.1-mini": {
        "input": 0.150,
        "output": 0.600,
    },
    "openai:gpt-4.1-small": {
        "input": 0.200,
        "output": 0.600,
    },
    "openai:gpt-4.1-large": {
        "input": 1.00,
        "output": 3.00,
    },

    # Assistant-responses versions
    "openai:gpt-4.1-responses": {
        "input": 0.50,
        "output": 1.50,
    },
    "openai:gpt-4.1-mini-responses": {
        "input": 0.150,
        "output": 0.600,
    },
    "openai:gpt-4.1-small-responses": {
        "input": 0.200,
        "output": 0.600,
    },
    "openai:gpt-4.1-large-responses": {
        "input": 1.00,
        "output": 3.00,
    },

    # ---------------------------------------------------------
    # GPT-4o FAMILY (4th Generation Omni)
    # ---------------------------------------------------------
    "openai:gpt-4o": {
        "input": 0.005,
        "output": 0.015,
    },
    "openai:gpt-4o-mini": {
        "input": 0.00015,
        "output": 0.00060,
    },
    "openai:gpt-4o-realtime": {
        "input": 0.005,
        "output": 0.015,
    },
    "openai:gpt-4o-audio-transcribe": {
        "input": 0.004,        # $0.004 / minute (but mapped as tokens)
        "output": 0.004,
    },

    # Vision / Vision-Preview models
    "openai:gpt-4o-vision": {
        "input": 0.005,
        "output": 0.015,
    },
    "openai:gpt-4o-mini-vision": {
        "input": 0.00015,
        "output": 0.00060,
    },
}
