"""
Middleware for tracking LLM token usage and costs.

This module provides an `@after_model` middleware hook that:
- Reads usage_metadata from the last AI message
- Computes the cost using MODEL_PRICES_USD configuration
- Prints per-call usage and a running total for the whole process
- Stores usage in session state for UI display (if streamlit is available)
"""

from typing import Dict, Any, Optional

from langchain.agents.middleware import after_model
from langchain_core.messages import AIMessage

from src.utils.config import MODEL_PRICES_USD
from src.utils.llm_utils import get_model_name_from_message

# Module-level storage for usage stats when runtime.context is None
_usage_storage: Dict[int, Dict[str, Any]] = {}

# Try to import streamlit for session state access
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None


@after_model
def log_llm_usage(state, runtime):
    """
    Middleware hook that runs after each model call.
    
    - Reads token usage from the last AIMessage
    - Computes per-call cost (if pricing is configured)
    - Maintains running totals in runtime.context.usage
    
    Args:
        state: The agent state containing messages
        runtime: The runtime context for storing usage totals
        
    Returns:
        None (middleware should not modify state)
    """
    if not state.get("messages"):
        return None

    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage):
        return None

    # usage_metadata is standardized across providers:
    # { "input_tokens": int, "output_tokens": int, "total_tokens": int, ... }
    usage = getattr(last_msg, "usage_metadata", None) or {}
    
    if not usage:
        # Some providers may not report usage yet
        print("[LLM USAGE] no usage_metadata available for this call")
        return None

    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens))

    model_name = get_model_name_from_message(last_msg)

    # Get or create running totals
    # Try to use runtime.context first, fallback to module-level storage
    usage_state = None
    
    if runtime.context is not None:
        # Try to get existing usage state from runtime.context
        usage_state = getattr(runtime.context, "usage", None)
        if usage_state is None:
            # Initialize usage state in runtime.context
            usage_state = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
            try:
                runtime.context.usage = usage_state
            except (AttributeError, TypeError):
                # If we can't set attributes on context, use fallback storage
                usage_state = None
    
    # Fallback to module-level storage if runtime.context is None or unusable
    if usage_state is None:
        runtime_id = id(runtime)
        if runtime_id not in _usage_storage:
            _usage_storage[runtime_id] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
        usage_state = _usage_storage[runtime_id]

    # Calculate cost first (needed for display)
    pricing = MODEL_PRICES_USD.get(model_name)
    call_cost = 0.0
    if pricing:
        call_cost = (
            (input_tokens / 1000.0) * pricing["input"]
            + (output_tokens / 1000.0) * pricing["output"]
        )

    # Update usage state
    usage_state["input_tokens"] += input_tokens
    usage_state["output_tokens"] += output_tokens
    usage_state["total_tokens"] += total_tokens
    usage_state["cost_usd"] += call_cost

    # Also update Streamlit session state if available
    if STREAMLIT_AVAILABLE:
        try:
            if "llm_usage" not in st.session_state:
                st.session_state.llm_usage = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                }
            
            st.session_state.llm_usage["input_tokens"] += input_tokens
            st.session_state.llm_usage["output_tokens"] += output_tokens
            st.session_state.llm_usage["total_tokens"] += total_tokens
            st.session_state.llm_usage["cost_usd"] += call_cost
        except (RuntimeError, AttributeError):
            # Streamlit not in context or session state not available
            pass

    # Pretty print for this single call
    if pricing:
        print(
            f"[LLM COST] model={model_name} "
            f"in={input_tokens} out={output_tokens} total={total_tokens} "
            f"call_cost=${call_cost:.6f} "
            f"(running_total=${usage_state['cost_usd']:.6f})"
        )
    else:
        print(
            f"[LLM TOKENS] model={model_name} "
            f"in={input_tokens} out={output_tokens} total={total_tokens} "
            f"(no pricing configured)"
        )

    # Return None to avoid modifying state
    return None


