"""Utility module for counting tokens in text."""

import json
from typing import Any, Optional

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        model: The model to use for tokenization (default: "gpt-4")
        
    Returns:
        Number of tokens in the text
    """
    if not text:
        return 0
    
    if not TIKTOKEN_AVAILABLE:
        # Fallback: approximate token count (rough estimate: 1 token ≈ 4 characters)
        return len(text) // 4
    
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(str(text)))
    except (KeyError, ValueError):
        # If model not found, use cl100k_base encoding (used by GPT-4)
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(str(text)))
        except Exception:
            # Final fallback: approximate count
            return len(text) // 4


def count_tokens_in_result(result: Any, model: str = "gpt-4") -> int:
    """
    Count tokens in a tool result (handles various data types).
    
    Args:
        result: The tool result (can be dict, list, string, etc.)
        model: The model to use for tokenization (default: "gpt-4")
        
    Returns:
        Number of tokens in the result
    """
    if result is None:
        return 0
    
    # Convert result to string representation
    if isinstance(result, (dict, list)):
        try:
            text = json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(result)
    else:
        text = str(result)
    
    return count_tokens(text, model)

