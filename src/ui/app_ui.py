import streamlit as st
import asyncio
import json
from src.agents.seo_agent import SEOAgent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from src.utils.tool_usage_tracker import get_tracker

# Try to import OpenAI error classes - they may be in different locations
try:
    from openai import (
        OpenAIError,
        APIError,
        AuthenticationError,
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
    )
except ImportError:
    try:
        from openai.error import (
            OpenAIError,
            APIError,
            AuthenticationError,
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
        )
    except ImportError:
        # Fallback: define base exception classes if OpenAI package structure is different
        class OpenAIError(Exception):
            pass
        class APIError(OpenAIError):
            pass
        class AuthenticationError(OpenAIError):
            pass
        class RateLimitError(OpenAIError):
            pass
        class APIConnectionError(OpenAIError):
            pass
        class APITimeoutError(OpenAIError):
            pass


def _is_openai_error(error: Exception) -> tuple[bool, str]:
    """
    Check if an error is an OpenAI-related error and return error type and message.
    Returns: (is_openai_error, error_type)
    """
    error_str = str(error).lower()
    error_type_str = type(error).__name__.lower()
    
    # Check for authentication errors
    if isinstance(error, AuthenticationError) or "authentication" in error_str or "api key" in error_str or "invalid api" in error_str:
        return True, "authentication"
    
    # Check for rate limit errors
    if isinstance(error, RateLimitError) or "rate limit" in error_str or "429" in error_str:
        return True, "rate_limit"
    
    # Check for timeout errors
    if isinstance(error, APITimeoutError) or "timeout" in error_str:
        return True, "timeout"
    
    # Check for connection errors
    if isinstance(error, APIConnectionError) or "connection" in error_str or "network" in error_str:
        return True, "connection"
    
    # Check for general API errors
    if isinstance(error, APIError) or isinstance(error, OpenAIError) or "openai" in error_type_str:
        return True, "api"
    
    return False, "unknown"


class AppUI:    
    def __init__(self):
        self.title = "🔍 SEO MCP Agent Chat"
        self.description = "Chat with your SEO expert agent powered by DataForSEO and Google Search Console"
        
        # Initialize agent in session state if not already initialized
        if "seo_agent" not in st.session_state:
            try:
                model_name = st.session_state.get("selected_model", "gpt-4o-mini")
                llm = ChatOpenAI(model=model_name, temperature=0, timeout=120*60) # 2 hrs
                st.session_state.seo_agent = SEOAgent(llm=llm)
            except AuthenticationError as e:
                st.error(f"🔐 **Authentication Error**: Invalid OpenAI API key. Please check your API key in the environment variables.\n\nError details: {str(e)}")
                st.stop()
            except APIError as e:
                st.error(f"⚠️ **OpenAI API Error**: {str(e)}\n\nPlease check your API configuration and try again.")
                st.stop()
            except Exception as e:
                st.error(f"❌ **Error initializing agent**: {str(e)}\n\nPlease check your configuration and try again.")
                st.stop()
        
    def _convert_messages_to_langchain(self, messages):
        """Convert Streamlit messages to LangChain message format."""
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        return langchain_messages
    
    async def _process_message(self, user_message: str):
        """Process a user message and return the agent's response."""
        try:
            # Convert messages to LangChain format
            langchain_messages = self._convert_messages_to_langchain(st.session_state.messages)
            langchain_messages.append(HumanMessage(content=user_message))
            
            # Run the agent
            result = await st.session_state.seo_agent.run(langchain_messages)
            
            # Extract the assistant's response
            if result and "messages" in result:
                assistant_message = result["messages"][-1]
                return assistant_message.content
            return "I apologize, but I couldn't generate a response. Please try again."
        except AuthenticationError as e:
            raise Exception(f"🔐 **Authentication Error**: Invalid OpenAI API key. Please check your API key in the environment variables.\n\nError details: {str(e)}")
        except RateLimitError as e:
            raise Exception(f"⏱️ **Rate Limit Error**: You've exceeded your OpenAI API rate limit. Please wait a moment and try again.\n\nError details: {str(e)}")
        except APITimeoutError as e:
            raise Exception(f"⏰ **Timeout Error**: The request to OpenAI timed out. Please try again with a simpler query.\n\nError details: {str(e)}")
        except APIConnectionError as e:
            raise Exception(f"🌐 **Connection Error**: Unable to connect to OpenAI API. Please check your internet connection.\n\nError details: {str(e)}")
        except APIError as e:
            raise Exception(f"⚠️ **OpenAI API Error**: {str(e)}\n\nPlease try again or check your API configuration.")
        except Exception as e:
            # Check if it's an OpenAI error by examining the error
            is_openai_err, error_type = _is_openai_error(e)
            
            if is_openai_err:
                if error_type == "authentication":
                    raise Exception(f"🔐 **Authentication Error**: Invalid OpenAI API key. Please check your API key in the environment variables.\n\nError details: {str(e)}")
                elif error_type == "rate_limit":
                    raise Exception(f"⏱️ **Rate Limit Error**: You've exceeded your OpenAI API rate limit. Please wait a moment and try again.\n\nError details: {str(e)}")
                elif error_type == "timeout":
                    raise Exception(f"⏰ **Timeout Error**: The request to OpenAI timed out. Please try again with a simpler query.\n\nError details: {str(e)}")
                elif error_type == "connection":
                    raise Exception(f"🌐 **Connection Error**: Unable to connect to OpenAI API. Please check your internet connection.\n\nError details: {str(e)}")
                else:
                    raise Exception(f"⚠️ **OpenAI API Error**: {str(e)}\n\nPlease try again or check your API configuration.")
            elif "**" in str(e):
                # Already a formatted error message
                raise
            else:
                raise Exception(f"❌ **Error**: {str(e)}\n\nPlease try again or contact support if the issue persists.")
    
    async def _stream_message(self, user_message: str, thinking_placeholder, response_placeholder):
        """Stream agent responses and update the UI in real-time with thinking process."""
        try:
            # Convert messages to LangChain format
            langchain_messages = self._convert_messages_to_langchain(st.session_state.messages)
            langchain_messages.append(HumanMessage(content=user_message))
            
            full_response = ""
            thinking_steps = []
            tool_calls_shown = set()
            tool_call_to_step = {}  # Map tool_call_id to step index
            
            # Stream the agent's response
            async for chunk in st.session_state.seo_agent.stream(langchain_messages):
                messages_in_chunk = chunk.get("messages", [])
                if not messages_in_chunk:
                    continue

                for current_message in messages_in_chunk:
                    # Handle tool calls - check if message has tool_calls attribute
                    if hasattr(current_message, "tool_calls") and current_message.tool_calls:
                        for tool_call in current_message.tool_calls:
                            tool_call_id = (
                                tool_call.get("id", "")
                                if isinstance(tool_call, dict)
                                else getattr(tool_call, "id", "")
                            )
                            if tool_call_id and tool_call_id not in tool_calls_shown:
                                tool_calls_shown.add(tool_call_id)
                                tool_name = (
                                    tool_call.get("name", "unknown")
                                    if isinstance(tool_call, dict)
                                    else getattr(tool_call, "name", "unknown")
                                )
                                tool_args = (
                                    tool_call.get("args", {})
                                    if isinstance(tool_call, dict)
                                    else getattr(tool_call, "args", {})
                                )

                                # Create a new thinking step for this tool call
                                step = {
                                    "type": "tool_call",
                                    "tool": tool_name,
                                    "args": tool_args,
                                    "status": "calling",
                                    "result": None,
                                }
                                thinking_steps.append(step)
                                tool_call_to_step[tool_call_id] = len(thinking_steps) - 1

                                # Update thinking display
                                self._update_thinking_display(
                                    thinking_placeholder, thinking_steps
                                )

                    # Handle tool results (ToolMessage)
                    if hasattr(current_message, "type") and current_message.type == "tool":
                        tool_call_id = (
                            getattr(current_message, "tool_call_id", None)
                            if hasattr(current_message, "tool_call_id")
                            else None
                        )
                        if tool_call_id and tool_call_id in tool_call_to_step:
                            step_idx = tool_call_to_step[tool_call_id]
                            thinking_steps[step_idx]["status"] = "completed"
                            # Store full tool result without truncation
                            tool_result = (
                                current_message.content
                                if hasattr(current_message, "content")
                                else str(current_message)
                            )
                            thinking_steps[step_idx]["result"] = tool_result
                            self._update_thinking_display(
                                thinking_placeholder, thinking_steps
                            )
                            
                            # Update sidebar stats in real-time when tool completes
                            # This happens after token counting in the middleware
                            # Note: Streamlit sidebar updates during streaming may be limited,
                            # but we try to update it here for real-time feedback
                            if "stats_placeholder" in st.session_state:
                                try:
                                    # Force update the stats display
                                    self._display_tool_usage_stats(st.session_state.stats_placeholder)
                                except Exception as e:
                                    # Log error but don't interrupt streaming
                                    # Sidebar updates during streaming can be unreliable in Streamlit
                                    pass

                    # Handle content messages (final AI response)
                    if hasattr(current_message, "content") and current_message.content:
                        # Only show content if it's from an AI message (not tool message)
                        message_type = getattr(current_message, "type", None)
                        if message_type == "ai" or (
                            not message_type
                            and not hasattr(current_message, "tool_call_id")
                        ):
                            full_response = current_message.content
                            response_placeholder.markdown(full_response)
            
            # Store thinking steps for later display
            if thinking_steps:
                st.session_state[f"thinking_{len(st.session_state.messages)}"] = thinking_steps
            
            # Final update of sidebar stats after streaming completes
            if "stats_placeholder" in st.session_state:
                try:
                    self._display_tool_usage_stats(st.session_state.stats_placeholder)
                except Exception:
                    pass
            
            # Return the final response and thinking steps
            return full_response if full_response else "I apologize, but I couldn't generate a response. Please try again.", thinking_steps
        except AuthenticationError as e:
            error_msg = f"🔐 **Authentication Error**: Invalid OpenAI API key. Please check your API key in the environment variables.\n\nError details: {str(e)}"
            response_placeholder.error(error_msg)
            return error_msg, []
        except RateLimitError as e:
            error_msg = f"⏱️ **Rate Limit Error**: You've exceeded your OpenAI API rate limit. Please wait a moment and try again.\n\nError details: {str(e)}"
            response_placeholder.error(error_msg)
            return error_msg, []
        except APITimeoutError as e:
            error_msg = f"⏰ **Timeout Error**: The request to OpenAI timed out. Please try again with a simpler query.\n\nError details: {str(e)}"
            response_placeholder.error(error_msg)
            return error_msg, []
        except APIConnectionError as e:
            error_msg = f"🌐 **Connection Error**: Unable to connect to OpenAI API. Please check your internet connection.\n\nError details: {str(e)}"
            response_placeholder.error(error_msg)
            return error_msg, []
        except APIError as e:
            error_msg = f"⚠️ **OpenAI API Error**: {str(e)}\n\nPlease try again or check your API configuration."
            response_placeholder.error(error_msg)
            return error_msg, []
        except Exception as e:
            # Check if it's an OpenAI error by examining the error
            is_openai_err, error_type = _is_openai_error(e)
            
            if is_openai_err:
                if error_type == "authentication":
                    error_msg = f"🔐 **Authentication Error**: Invalid OpenAI API key. Please check your API key in the environment variables.\n\nError details: {str(e)}"
                elif error_type == "rate_limit":
                    error_msg = f"⏱️ **Rate Limit Error**: You've exceeded your OpenAI API rate limit. Please wait a moment and try again.\n\nError details: {str(e)}"
                elif error_type == "timeout":
                    error_msg = f"⏰ **Timeout Error**: The request to OpenAI timed out. Please try again with a simpler query.\n\nError details: {str(e)}"
                elif error_type == "connection":
                    error_msg = f"🌐 **Connection Error**: Unable to connect to OpenAI API. Please check your internet connection.\n\nError details: {str(e)}"
                else:
                    error_msg = f"⚠️ **OpenAI API Error**: {str(e)}\n\nPlease try again or check your API configuration."
            elif "**" in str(e):
                # Already a formatted error message
                error_msg = str(e)
            else:
                error_msg = f"❌ **Error**: {str(e)}\n\nPlease try again or contact support if the issue persists."
            
            response_placeholder.error(error_msg)
            return error_msg, []
    
    def _update_thinking_display(self, placeholder, thinking_steps):
        """Update the thinking process display with all steps during streaming."""
        if not thinking_steps:
            placeholder.empty()
            return
        
        thinking_html = "<div style='font-size: 0.9em; color: #666;'>"
        thinking_html += "<strong>🤔 Agent Thinking Process:</strong><br><br>"
        
        for i, step in enumerate(thinking_steps, 1):
            if step["type"] == "tool_call":
                thinking_html += f"<div style='margin-bottom: 10px; padding: 8px; background-color: #f0f0f0; border-radius: 5px;'>"
                thinking_html += f"<strong>Step {i}:</strong> Calling tool <code>{step['tool']}</code><br>"
                
                if step.get("args"):
                    args_str = ", ".join([f"{k}={v}" for k, v in list(step["args"].items())[:3]])
                    if len(step["args"]) > 3:
                        args_str += "..."
                    thinking_html += f"<small>Arguments: {args_str}</small><br>"
                
                if step.get("status") == "completed":
                    thinking_html += f"<span style='color: green;'>✓ Completed</span>"
                    if step.get("result"):
                        # Show full result in streaming display (can be long, but user can scroll)
                        result_text = step["result"]
                        # For very long results, show a preview but indicate full content is available
                        if len(result_text) > 500:
                            result_preview = result_text[:500] + f"... (showing preview, {len(result_text)} chars total - see full result below)"
                        else:
                            result_preview = result_text
                        thinking_html += f"<br><small style='color: #555;'>Result: {result_preview}</small>"
                else:
                    thinking_html += f"<span style='color: orange;'>⏳ In progress...</span>"
                
                thinking_html += "</div>"
        
        thinking_html += "</div>"
        placeholder.markdown(thinking_html, unsafe_allow_html=True)
    
    def _display_thinking_steps(self, thinking_steps):
        """Display thinking steps in a formatted way in the expander."""
        if not thinking_steps:
            st.info("No thinking steps captured.")
            return
        
        st.markdown(f"**Total Steps:** {len(thinking_steps)}")
        st.markdown("---")
        
        for i, step in enumerate(thinking_steps, 1):
            if step["type"] == "tool_call":
                # Create a card-like container for each step
                with st.container():
                    # Header with step number and tool name
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**Step {i}:** `{step['tool']}`")
                    with col2:
                        if step.get("status") == "completed":
                            st.success("✓ Completed")
                        else:
                            st.warning("⏳ In progress")
                    
                    # Display arguments
                    if step.get("args"):
                        with st.expander("📋 View Arguments", expanded=False):
                            args_display = {k: str(v)[:200] if len(str(v)) > 200 else v 
                                          for k, v in list(step["args"].items())[:10]}
                            if len(step["args"]) > 10:
                                args_display["..."] = f"{len(step['args']) - 10} more arguments"
                            st.json(args_display)
                    
                    # Display result if available
                    if step.get("result"):
                        result_text = step["result"]
                        with st.expander(f"📊 View Result ({len(result_text)} chars)", expanded=False):
                            # Try to format as JSON if possible, otherwise show as text
                            try:
                                parsed_result = json.loads(result_text)
                                st.json(parsed_result)
                            except (json.JSONDecodeError, TypeError):
                                # Use a larger height and allow scrolling for full content
                                st.text_area(
                                    "Result:",
                                    value=result_text,
                                    height=400,
                                    disabled=True,
                                    label_visibility="collapsed",
                                    key=f"thinking_result_{i}",
                                )
                    
                    if i < len(thinking_steps):
                        st.divider()
    
    def _run_async(self, coro):
        """Helper to run async functions in Streamlit."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    
    def _display_tool_usage_stats(self, placeholder=None):
        """Display tool usage statistics in a dropdown/expander."""
        tracker = get_tracker()
        usage_stats = tracker.get_sorted_stats(reverse=True)
        
        # Only show if there are tools that have been used
        if not usage_stats:
            if placeholder:
                placeholder.empty()
            return
        
        total_calls = tracker.get_total_calls()
        total_tokens = tracker.get_total_tokens()
        
        # Use placeholder if provided, otherwise create new expander
        if placeholder:
            # Clear and recreate the content in the placeholder for real-time updates
            placeholder.empty()
            with placeholder.container():
                with st.expander("📊 Tool Usage Statistics", expanded=False):
                    self._render_stats_content(tracker, usage_stats, total_calls, total_tokens)
        else:
            with st.expander("📊 Tool Usage Statistics", expanded=False):
                self._render_stats_content(tracker, usage_stats, total_calls, total_tokens)
    
    def _render_stats_content(self, tracker, usage_stats, total_calls, total_tokens):
        """Render the stats content (separated for reuse)."""
        # Summary section
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Tool Calls", total_calls)
        with col2:
            st.metric("Total Tokens", f"{total_tokens:,}" if total_tokens > 0 else "0")
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2 = st.tabs(["📈 Usage Count", "🔤 Token Usage"])
        
        with tab1:
            st.caption("Tool usage by call count")
            # Display stats in a table format
            for tool_name, count in usage_stats.items():
                # Calculate percentage
                percentage = (count / total_calls) * 100 if total_calls > 0 else 0
                
                # Create a progress bar-like display
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{tool_name}**")
                with col2:
                    st.markdown(f"`{count}`")
                with col3:
                    st.markdown(f"_{percentage:.1f}%_")
                
                # Add a visual progress bar
                st.progress(percentage / 100)
        
        with tab2:
            st.caption("Tool usage by token output")
            token_stats = tracker.get_sorted_token_stats(reverse=True)
            
            if token_stats:
                for tool_name, token_count in token_stats.items():
                    # Get detailed token stats
                    total_tokens_tool, avg_tokens, min_tokens, max_tokens = tracker.get_token_stats(tool_name)
                    tool_calls = tracker.get_usage_count(tool_name)
                    
                    # Calculate percentage of total tokens
                    token_percentage = (token_count / total_tokens) * 100 if total_tokens > 0 else 0
                    
                    # Display token information
                    col1, col2 = st.columns([3, 2])
                    with col1:
                        st.markdown(f"**{tool_name}**")
                        st.caption(f"Total: {token_count:,} tokens | Avg: {avg_tokens:.0f} | Range: {min_tokens:,} - {max_tokens:,}")
                    with col2:
                        st.markdown(f"`{token_count:,}` tokens")
                        st.markdown(f"_{token_percentage:.1f}%_ of total")
                    
                    # Add a visual progress bar
                    st.progress(token_percentage / 100)
            else:
                st.info("No token data available yet. Token counts are tracked after tool execution.")
        
        # Add reset button
        st.markdown("---")
        if st.button("🔄 Reset Statistics", use_container_width=True, key="reset_stats_btn"):
            tracker.reset_stats()
            st.rerun()
    
    def run(self):
        """Main Streamlit app run method."""
        st.title(self.title)
        st.markdown(self.description)
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            model_options = [
                "gpt-5.1",
                "gpt-5",
                "gpt-5-mini",
                "gpt-4.1","gpt-4o-mini", "gpt-4o"]
            default_index = model_options.index(
                st.session_state.get("selected_model", "gpt-5.1")
            )
            selected_model = st.selectbox(
                "Model",
                model_options,
                index=default_index,
                key="model_selector"
            )
            
            # Reinitialize agent if model changed
            if st.session_state.get("selected_model") != selected_model:
                try:
                    st.session_state.selected_model = selected_model
                    llm = ChatOpenAI(model=selected_model, temperature=0, timeout=120*60)
                    st.session_state.seo_agent = SEOAgent(llm=llm)
                    st.success(f"✅ Model changed to {selected_model}")
                except AuthenticationError as e:
                    st.error(f"🔐 **Authentication Error**: Invalid OpenAI API key. Please check your API key in the environment variables.\n\nError details: {str(e)}")
                except APIError as e:
                    st.error(f"⚠️ **OpenAI API Error**: {str(e)}\n\nPlease check your API configuration and try again.")
                except Exception as e:
                    st.error(f"❌ **Error changing model**: {str(e)}\n\nPlease check your configuration and try again.")
            
            st.divider()
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
            
            # Display tool usage statistics (only when tools have been used)
            st.divider()
            # Create a placeholder for stats that can be updated in real-time
            if "stats_placeholder" not in st.session_state:
                st.session_state.stats_placeholder = st.empty()
            self._display_tool_usage_stats(st.session_state.stats_placeholder)

        # Initialize messages in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask me anything about SEO"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Show assistant response with streaming
            with st.chat_message("assistant"):
                # Create placeholders for thinking and response
                thinking_placeholder = st.empty()
                response_placeholder = st.empty()
                
                try:
                    # Stream the message asynchronously
                    result = self._run_async(
                        self._stream_message(prompt, thinking_placeholder, response_placeholder)
                    )
                    
                    # Unpack result (response, thinking_steps)
                    if isinstance(result, tuple):
                        response, thinking_steps = result
                    else:
                        response = result
                        thinking_steps = []
                    
                    # After streaming completes, show final layout
                    if response:
                        # Clear thinking placeholder
                        thinking_placeholder.empty()

                        # Check if response is an error message (starts with error emoji or contains **)
                        is_error = response.startswith(("🔐", "⏱️", "⏰", "🌐", "⚠️", "❌")) or "**" in response
                        
                        if not is_error:
                            tool_warning = getattr(st.session_state.seo_agent, "get_tool_warning", None)
                            warning_message = tool_warning() if callable(tool_warning) else None
                            if warning_message:
                                st.warning(warning_message)
                            
                            # Show thinking process in an expander if there are steps
                            if thinking_steps:
                                with st.expander("🤔 View Agent Thinking Process", expanded=False):
                                    st.caption("See the tools and reasoning steps the agent used to generate this response")
                                    self._display_thinking_steps(thinking_steps)
                            
                            # Show final response prominently with a divider
                            if thinking_steps:
                                st.divider()
                            response_placeholder.markdown(response)
                        else:
                            # Error message already displayed in _stream_message
                            pass
                    
                    # Add assistant response to chat history (even if it's an error)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    # Fallback error handling if _stream_message doesn't catch it
                    error_msg = f"❌ **Unexpected Error**: {str(e)}\n\nPlease try again or contact support if the issue persists."
                    response_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})