"""Tool error handling middleware for LangChain agents with timing & timeout."""

import asyncio
import json
import os
import time
from typing import Any, Optional

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from src.utils.tool_usage_tracker import get_tracker
from src.utils.token_counter import count_tokens_in_result
from src.utils.tool_output_logger import get_logger
from src.utils.lighthouse_transformer import extract_lighthouse_seo_summary


def _get_timeout_seconds() -> Optional[float]:
    """Read the tool timeout (in seconds) from env (defaults to 120s)."""
    value = os.getenv("TOOL_TIMEOUT_SECONDS", "120").strip()
    if not value:
        return None
    try:
        timeout = float(value)
    except ValueError:
        return 120.0
    return timeout if timeout > 0 else None


TOOL_TIMEOUT_SECONDS = _get_timeout_seconds()
LOG_TOOL_TIMINGS = os.getenv("LOG_TOOL_TIMINGS", "true").lower() in ("1", "true", "yes")
LOG_TOOL_OUTPUT = os.getenv("LOG_TOOL_OUTPUT", "true").lower() in ("1", "true", "yes")
LOG_TOOL_TOKENS = os.getenv("LOG_TOOL_TOKENS", "true").lower() in ("1", "true", "yes")


def _log_tool_timing(tool_name: str, duration: float, status: str) -> None:
    """Print a simple timing log for observability."""
    if not LOG_TOOL_TIMINGS:
        return
    tool_label = tool_name or "unknown_tool"
    print(f"[tool:{tool_label}] {status} after {duration:.2f}s")


def _serialize_result(result: Any) -> str:
    """Convert tool result into a JSON string when possible."""
    if isinstance(result, (dict, list)):
        try:
            return json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError):
            pass
    return str(result)


def _log_tool_output(tool_name: str, result: Any, limit: int = 100) -> None:
    """Log the tool output, truncated to `limit` characters."""
    if not LOG_TOOL_OUTPUT:
        return
    serialized = _serialize_result(result)
    truncated = serialized[:limit]
    suffix = "…" if len(serialized) > limit else ""
    tool_label = tool_name or "unknown_tool"
    print(f"[tool:{tool_label}] output: {truncated}{suffix}")


def _log_tool_tokens(tool_name: str, token_count: int) -> None:
    """Log the token count for a tool output."""
    if not LOG_TOOL_TOKENS:
        return
    tool_label = tool_name or "unknown_tool"
    print(f"[tool:{tool_label}] output tokens: {token_count}")


def _transform_lighthouse_result(result: Any) -> Any:
    """
    Transform on_page_lighthouse tool result to SEO-focused summary.
    
    Handles different result formats:
    - ToolMessage with content attribute (string or dict)
    - Direct dict with 'data' key (wrapped format from logger)
    - Direct dict with 'items' key (raw lighthouse format)
    - String that can be parsed as JSON
    
    Returns the transformed result, or original if transformation fails.
    """
    try:
        # Extract the actual data from the result
        data_to_transform = None
        is_tool_message = False
        tool_call_id = None
        
        # If it's a ToolMessage, get the content and preserve tool_call_id
        if isinstance(result, ToolMessage):
            is_tool_message = True
            tool_call_id = result.tool_call_id
            content = result.content
            # Try to parse content as JSON if it's a string
            if isinstance(content, str):
                try:
                    data_to_transform = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    # If parsing fails, return original
                    return result
            elif isinstance(content, (dict, list)):
                data_to_transform = content
            else:
                return result
        # If it's already a dict/list, use it directly
        elif isinstance(result, (dict, list)):
            data_to_transform = result
        else:
            # Unknown format, return original
            return result
        
        # Ensure we have a dict to transform
        if not isinstance(data_to_transform, dict):
            return result
        
        # Handle different input formats:
        # 1. Wrapped format: {"tool_name": "...", "data": {...}}
        # 2. Raw lighthouse format: {"items": [...], "audits": {...}, ...}
        # 3. Direct data format: {"data": {"items": [...]}}
        
        # Check if it's already in the format expected by extract_lighthouse_seo_summary
        # (has 'data' key with 'items' inside)
        if "data" in data_to_transform and isinstance(data_to_transform["data"], dict):
            # This is the wrapped format, use it directly
            payload = data_to_transform
        elif "items" in data_to_transform:
            # This is raw lighthouse format, wrap it
            payload = {"data": data_to_transform}
        elif isinstance(data_to_transform.get("data"), dict) and "items" in data_to_transform.get("data", {}):
            # Already in correct format
            payload = data_to_transform
        else:
            # Unknown format, return original
            return result
        
        # Transform the data
        transformed = extract_lighthouse_seo_summary(payload)
        
        # Return in the same format as input
        if is_tool_message:
            return ToolMessage(
                content=json.dumps(transformed, ensure_ascii=False),
                tool_call_id=tool_call_id,
            )
        # Otherwise return the transformed dict
        return transformed
        
    except Exception as e:
        # If transformation fails, log warning but return original
        print(f"Warning: Failed to transform lighthouse result: {e}")
        import traceback
        traceback.print_exc()
        return result


@wrap_tool_call
async def handle_tool_errors(request, handler):
    """
    Handle tool execution errors with custom messages (async version).

    Adds:
    - Per-tool timeout via TOOL_TIMEOUT_SECONDS env (default 120s).
    - Simple timing logs to help diagnose long-running tools.

    Args:
        request: The tool call request object
        handler: The async handler function to execute the tool call

    Returns:
        ToolMessage | Any: ToolMessage on error/timeout, otherwise handler result.
    """
    start_time = time.perf_counter()
    tool_name = request.tool_call.get("name") if request.tool_call else "unknown_tool"
    tool_input = request.tool_call.get("args", {}) if request.tool_call else {}

    # Track tool usage
    tracker = get_tracker()

    try:
        if TOOL_TIMEOUT_SECONDS is not None:
            result = await asyncio.wait_for(handler(request), timeout=TOOL_TIMEOUT_SECONDS)
        else:
            result = await handler(request)

        # Transform result for on_page_lighthouse tool
        if tool_name == "on_page_lighthouse":
            result = _transform_lighthouse_result(result)

        # Count tokens in the result
        token_count = count_tokens_in_result(result)
        
        # Track tool usage with token count
        tracker.track_tool_call(tool_name, token_count)

        # Log tool output to file (JSON or markdown) with token count and input
        logger = get_logger()
        logger.log_tool_output(tool_name, result, token_count, tool_input)

        duration = time.perf_counter() - start_time
        _log_tool_timing(tool_name, duration, "completed")
        _log_tool_output(tool_name, result)
        _log_tool_tokens(tool_name, token_count)
        return result

    except asyncio.TimeoutError:
        duration = time.perf_counter() - start_time
        _log_tool_timing(tool_name, duration, "timed out")
        error_result = {
            "error": "timeout",
            "message": f"Tool timeout: `{tool_name}` did not finish within {TOOL_TIMEOUT_SECONDS:.0f} seconds",
            "duration": duration
        }
        # Log timeout error to file with input
        logger = get_logger()
        logger.log_tool_output(tool_name, error_result, None, tool_input)
        return ToolMessage(
            content=(
                f"Tool timeout: `{tool_name}` did not finish within "
                f"{TOOL_TIMEOUT_SECONDS:.0f} seconds. Please simplify the request "
                "or try again."
            ),
            tool_call_id=request.tool_call["id"],
        )
    except Exception as e:
        duration = time.perf_counter() - start_time
        _log_tool_timing(tool_name, duration, "failed")
        error_result = {
            "error": "exception",
            "message": str(e),
            "error_type": type(e).__name__,
            "duration": duration
        }
        # Log error to file with input
        logger = get_logger()
        logger.log_tool_output(tool_name, error_result, None, tool_input)
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"],
        )

