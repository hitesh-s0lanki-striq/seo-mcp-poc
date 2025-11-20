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

    # Track tool usage
    tracker = get_tracker()

    try:
        if TOOL_TIMEOUT_SECONDS is not None:
            result = await asyncio.wait_for(handler(request), timeout=TOOL_TIMEOUT_SECONDS)
        else:
            result = await handler(request)

        # Count tokens in the result
        token_count = count_tokens_in_result(result)
        
        # Track tool usage with token count
        tracker.track_tool_call(tool_name, token_count)

        duration = time.perf_counter() - start_time
        _log_tool_timing(tool_name, duration, "completed")
        _log_tool_output(tool_name, result)
        _log_tool_tokens(tool_name, token_count)
        return result

    except asyncio.TimeoutError:
        duration = time.perf_counter() - start_time
        _log_tool_timing(tool_name, duration, "timed out")
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
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"],
        )

