"""Tool error handling middleware for LangChain agents."""

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage


@wrap_tool_call
async def handle_tool_errors(request, handler):
    """
    Handle tool execution errors with custom messages (async version).
    
    This middleware wraps tool calls and catches any exceptions that occur
    during tool execution. Instead of letting the error propagate, it returns
    a user-friendly ToolMessage that the agent can use to inform the user
    about the error.
    
    This async version is required when using ainvoke() or astream() methods.
    
    Args:
        request: The tool call request object
        handler: The async handler function to execute the tool call
        
    Returns:
        ToolMessage: A message containing the error information if an error occurs,
                     otherwise returns the result from the handler
    """
    try:
        # Await the handler since it's async
        return await handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

