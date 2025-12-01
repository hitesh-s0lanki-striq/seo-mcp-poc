"""
Utility functions for LLM cost analyser.
"""

from typing import Any, Dict

from langchain_core.messages import AIMessage


def get_model_name_from_message(msg: AIMessage) -> str:
    """
    Extract model name from an AIMessage.
    
    Tries multiple common locations where the model name might be stored:
    - response_metadata.model
    - response_metadata.model_name
    - msg.model attribute
    
    Args:
        msg: The AIMessage to extract model name from
        
    Returns:
        The model name as a string, or "unknown" if not found
    """
    # Try a few common places where model name might live
    rm: Dict[str, Any] = getattr(msg, "response_metadata", {}) or {}
    return (
        rm.get("model")
        or rm.get("model_name")
        or getattr(msg, "model", None)
        or "unknown"
    )


