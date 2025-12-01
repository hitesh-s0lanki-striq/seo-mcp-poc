import streamlit as st
import asyncio
import json
from src.agents.seo_agent import SEOAgent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from src.utils.tool_usage_tracker import get_tracker
from src.instructions.seo_agent_instruction import get_seo_agent_instructions
from src.utils.config import MODEL_PRICES_USD
from src.utils.llm_utils import get_model_name_from_message

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
        
        # Initialize default system prompt in session state if not already set
        if "system_prompt" not in st.session_state:
            st.session_state.system_prompt = get_seo_agent_instructions()
        
        # Initialize agent in session state if not already initialized
        if "seo_agent" not in st.session_state:
            try:
                model_name = st.session_state.get("selected_model", "gpt-4.1")
                llm = ChatOpenAI(model=model_name, temperature=0, timeout=120*60) # 2 hrs
                agent = SEOAgent(llm=llm)
                # Set the system prompt from session state
                agent.update_system_prompt(st.session_state.system_prompt)
                st.session_state.seo_agent = agent
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
    
    def _update_usage_from_message(self, message):
        """Extract usage metadata from an AI message and update session state."""
        if not isinstance(message, AIMessage):
            return
        
        usage = getattr(message, "usage_metadata", None) or {}
        if not usage:
            return
        
        # Initialize if not present
        if "llm_usage" not in st.session_state:
            st.session_state.llm_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
        
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens))
        
        model_name = get_model_name_from_message(message)
        pricing = MODEL_PRICES_USD.get(model_name)
        
        call_cost = 0.0
        if pricing:
            call_cost = (
                (input_tokens / 1000.0) * pricing["input"]
                + (output_tokens / 1000.0) * pricing["output"]
            )
        
        # Update session state
        st.session_state.llm_usage["input_tokens"] += input_tokens
        st.session_state.llm_usage["output_tokens"] += output_tokens
        st.session_state.llm_usage["total_tokens"] += total_tokens
        st.session_state.llm_usage["cost_usd"] += call_cost
        
        # Update the display if placeholder exists
        if "usage_placeholder" in st.session_state:
            try:
                self._display_usage_stats(st.session_state.usage_placeholder)
            except Exception:
                pass
    
    def _display_usage_stats(self, placeholder=None):
        """Display LLM usage statistics (tokens and cost) in the sidebar."""
        if "llm_usage" not in st.session_state:
            st.session_state.llm_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
        
        usage = st.session_state.llm_usage
        
        if placeholder:
            # Use container for real-time updates during streaming
            # Streamlit will update this during async operations
            with placeholder.container():
                st.header("📊 Usage Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Tokens", f"{usage['total_tokens']:,}")
                with col2:
                    st.metric("Total Cost", f"${usage['cost_usd']:.4f}")
                
                # Show breakdown in expander
                with st.expander("📈 Token Breakdown", expanded=False):
                    st.metric("Input Tokens", f"{usage['input_tokens']:,}")
                    st.metric("Output Tokens", f"{usage['output_tokens']:,}")
                    if usage['total_tokens'] > 0:
                        input_pct = (usage['input_tokens'] / usage['total_tokens']) * 100
                        output_pct = (usage['output_tokens'] / usage['total_tokens']) * 100
                        st.caption(f"Input: {input_pct:.1f}% | Output: {output_pct:.1f}%")
                
                # Reset button for usage stats
                if st.button("🔄 Reset Usage Stats", use_container_width=True, key="reset_usage_btn"):
                    st.session_state.llm_usage = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "cost_usd": 0.0,
                    }
                    st.rerun()
        else:
            # Fallback: display directly (for initial render)
            st.header("📊 Usage Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Tokens", f"{usage['total_tokens']:,}")
            with col2:
                st.metric("Total Cost", f"${usage['cost_usd']:.4f}")
            
            # Show breakdown in expander
            with st.expander("📈 Token Breakdown", expanded=False):
                st.metric("Input Tokens", f"{usage['input_tokens']:,}")
                st.metric("Output Tokens", f"{usage['output_tokens']:,}")
                if usage['total_tokens'] > 0:
                    input_pct = (usage['input_tokens'] / usage['total_tokens']) * 100
                    output_pct = (usage['output_tokens'] / usage['total_tokens']) * 100
                    st.caption(f"Input: {input_pct:.1f}% | Output: {output_pct:.1f}%")
            
            # Reset button for usage stats
            if st.button("🔄 Reset Usage Stats", use_container_width=True, key="reset_usage_btn"):
                st.session_state.llm_usage = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                }
                st.rerun()
    
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
    
    async def _stream_message(
        self,
        user_message: str,
        thinking_placeholder,
        response_placeholder,
        spinner_placeholder=None,
        prev_assistant_content: str | None = None,
    ):
        """Stream agent responses and update the UI in real-time with thinking process."""
        try:
            # Clear placeholders at the start to prevent showing previous content
            if response_placeholder:
                response_placeholder.empty()
            if thinking_placeholder:
                thinking_placeholder.empty()
            
            # Convert messages to LangChain format
            # Note: st.session_state.messages already includes the new user message,
            # so we don't need to append it again
            langchain_messages = self._convert_messages_to_langchain(st.session_state.messages)
            
            full_response = ""
            thinking_steps = []
            tool_calls_shown = set()
            tool_call_to_step = {}  # Map tool_call_id to step index
            spinner_shown = False
            processed_message_ids = set()  # Track processed messages to avoid double-counting usage
            
            # Stream the agent's response
            async for chunk in st.session_state.seo_agent.stream(langchain_messages):
                messages_in_chunk = chunk.get("messages", [])
                if not messages_in_chunk:
                    continue

                # ✅ Process ALL messages (so tool calls & tool results are seen),
                #    but we'll be smart about AI messages to avoid replaying old answers
                for current_message in messages_in_chunk:
                    # --- TOOL CALLS ---
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

                                step = {
                                    "type": "tool_call",
                                    "tool": tool_name,
                                    "args": tool_args,
                                    "status": "calling",
                                    "result": None,
                                }
                                thinking_steps.append(step)
                                tool_call_to_step[tool_call_id] = len(thinking_steps) - 1

                                if thinking_placeholder:
                                    self._update_thinking_display(
                                        thinking_placeholder, thinking_steps
                                    )

                    # --- TOOL RESULTS ---
                    if hasattr(current_message, "type") and current_message.type == "tool":
                        tool_call_id = (
                            getattr(current_message, "tool_call_id", None)
                            if hasattr(current_message, "tool_call_id")
                            else None
                        )
                        if tool_call_id and tool_call_id in tool_call_to_step:
                            step_idx = tool_call_to_step[tool_call_id]
                            thinking_steps[step_idx]["status"] = "completed"
                            tool_result = (
                                current_message.content
                                if hasattr(current_message, "content")
                                else str(current_message)
                            )
                            thinking_steps[step_idx]["result"] = tool_result

                            if thinking_placeholder:
                                self._update_thinking_display(
                                    thinking_placeholder, thinking_steps
                                )

                            if "stats_placeholder" in st.session_state:
                                try:
                                    self._display_tool_usage_stats(
                                        st.session_state.stats_placeholder
                                    )
                                except Exception:
                                    pass
                            
                            # Also update usage stats display when tool results are processed
                            if "usage_placeholder" in st.session_state:
                                try:
                                    self._display_usage_stats(st.session_state.usage_placeholder)
                                except Exception:
                                    pass

                    # --- AI CONTENT ---
                    if hasattr(current_message, "content") and current_message.content:
                        message_type = getattr(current_message, "type", None)
                        if message_type == "ai" or (
                            not message_type and not hasattr(current_message, "tool_call_id")
                        ):
                            # Update usage stats from this message (only once per message)
                            message_id = id(current_message)
                            if message_id not in processed_message_ids:
                                self._update_usage_from_message(current_message)
                                processed_message_ids.add(message_id)
                                
                                # Force immediate display update after usage update
                                if "usage_placeholder" in st.session_state:
                                    try:
                                        self._display_usage_stats(st.session_state.usage_placeholder)
                                    except Exception:
                                        pass
                            
                            content = current_message.content
                            
                            # ⛔ Skip AI messages that are just the previous turn's final answer
                            if prev_assistant_content and content == prev_assistant_content:
                                continue
                            
                            # ⛔ Skip if it's identical to what we already rendered in this turn
                            if content == full_response:
                                continue
                            
                            # ✅ Now it's genuinely new content for this turn
                            if spinner_placeholder and not spinner_shown:
                                spinner_placeholder.empty()
                                spinner_shown = True

                            full_response = content
                            if response_placeholder:
                                response_placeholder.markdown(full_response)

            # Process all messages one more time to ensure we capture all usage data
            # This helps catch any usage metadata we might have missed during streaming
            if "messages" in chunk:
                for msg in chunk["messages"]:
                    if isinstance(msg, AIMessage):
                        message_id = id(msg)
                        if message_id not in processed_message_ids:
                            self._update_usage_from_message(msg)
                            processed_message_ids.add(message_id)
            
            # Final update of sidebar stats after streaming completes
            if "stats_placeholder" in st.session_state:
                try:
                    self._display_tool_usage_stats(st.session_state.stats_placeholder)
                except Exception:
                    pass
            
            # Final update of usage stats after streaming completes
            if "usage_placeholder" in st.session_state:
                try:
                    self._display_usage_stats(st.session_state.usage_placeholder)
                except Exception:
                    pass
            
            # Clear spinner if it's still showing
            if spinner_placeholder and not spinner_shown:
                spinner_placeholder.empty()
            
            # Return the final response and thinking steps
            if not full_response:
                full_response = "I apologize, but I couldn't generate a response. Please try again."
            return full_response, thinking_steps

        except AuthenticationError as e:
            if spinner_placeholder:
                spinner_placeholder.empty()
            error_msg = f"🔐 **Authentication Error**: Invalid OpenAI API key. Please check your API key in the environment variables.\n\nError details: {str(e)}"
            if response_placeholder:
                response_placeholder.error(error_msg)
            return error_msg, []
        except RateLimitError as e:
            if spinner_placeholder:
                spinner_placeholder.empty()
            error_msg = f"⏱️ **Rate Limit Error**: You've exceeded your OpenAI API rate limit. Please wait a moment and try again.\n\nError details: {str(e)}"
            if response_placeholder:
                response_placeholder.error(error_msg)
            return error_msg, []
        except APITimeoutError as e:
            if spinner_placeholder:
                spinner_placeholder.empty()
            error_msg = f"⏰ **Timeout Error**: The request to OpenAI timed out. Please try again with a simpler query.\n\nError details: {str(e)}"
            if response_placeholder:
                response_placeholder.error(error_msg)
            return error_msg, []
        except APIConnectionError as e:
            if spinner_placeholder:
                spinner_placeholder.empty()
            error_msg = f"🌐 **Connection Error**: Unable to connect to OpenAI API. Please check your internet connection.\n\nError details: {str(e)}"
            if response_placeholder:
                response_placeholder.error(error_msg)
            return error_msg, []
        except APIError as e:
            if spinner_placeholder:
                spinner_placeholder.empty()
            error_msg = f"⚠️ **OpenAI API Error**: {str(e)}\n\nPlease try again or check your API configuration."
            if response_placeholder:
                response_placeholder.error(error_msg)
            return error_msg, []
        except Exception as e:
            # Clear spinner on error
            if spinner_placeholder:
                spinner_placeholder.empty()
            
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
                error_msg = str(e)
            else:
                error_msg = f"❌ **Error**: {str(e)}\n\nPlease try again or contact support if the issue persists."
            
            if response_placeholder:
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
            # Initialize LLM usage tracking in session state if not present
            if "llm_usage" not in st.session_state:
                st.session_state.llm_usage = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                }
            
            # Create placeholder for usage stats (for real-time updates)
            if "usage_placeholder" not in st.session_state:
                st.session_state.usage_placeholder = st.empty()
            
            # Display usage stats (will be updated in real-time during streaming)
            self._display_usage_stats(st.session_state.usage_placeholder)
            
            st.divider()
            
            st.header("⚙️ Configuration")
            model_options = [
                "gpt-5.1",
                "gpt-5",
                "gpt-5-mini",
                "gpt-4.1", "gpt-4o-mini", "gpt-4o"
            ]
            default_model = st.session_state.get("selected_model", "gpt-4.1")
            try:
                default_index = model_options.index(default_model)
            except ValueError:
                default_index = model_options.index("gpt-4.1")

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
                    agent = SEOAgent(llm=llm)
                    # Set the system prompt from session state
                    agent.update_system_prompt(st.session_state.system_prompt)
                    st.session_state.seo_agent = agent
                    st.success(f"✅ Model changed to {selected_model}")
                except AuthenticationError as e:
                    st.error(f"🔐 **Authentication Error**: Invalid OpenAI API key. Please check your API key in the environment variables.\n\nError details: {str(e)}")
                except APIError as e:
                    st.error(f"⚠️ **OpenAI API Error**: {str(e)}\n\nPlease check your API configuration and try again.")
                except Exception as e:
                    st.error(f"❌ **Error changing model**: {str(e)}\n\nPlease check your configuration and try again.")
            
            st.divider()
            
            # System Prompt Editor
            st.subheader("📝 System Prompt")
            with st.expander("Edit System Prompt", expanded=False):
                default_prompt = get_seo_agent_instructions()
                
                # Text area for editing system prompt
                edited_prompt = st.text_area(
                    "System Prompt:",
                    value=st.session_state.system_prompt,
                    height=300,
                    help="Customize the system prompt that guides the agent's behavior. The default is loaded from the instruction file.",
                    key="system_prompt_editor"
                )
                
                # Buttons for managing system prompt
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save & Apply", use_container_width=True, key="save_system_prompt"):
                        if edited_prompt.strip():
                            st.session_state.system_prompt = edited_prompt.strip()
                            # Update the agent's system prompt and invalidate cache
                            st.session_state.seo_agent.update_system_prompt(edited_prompt.strip())
                            st.success("✅ System prompt updated! The agent will use this prompt for new conversations.")
                        else:
                            st.error("❌ System prompt cannot be empty!")
                
                with col2:
                    if st.button("🔄 Reset to Default", use_container_width=True, key="reset_system_prompt"):
                        st.session_state.system_prompt = default_prompt
                        # Update the agent's system prompt and invalidate cache
                        st.session_state.seo_agent.update_system_prompt(default_prompt)
                        st.success("✅ System prompt reset to default!")
                        st.rerun()
                
                # Show character count
                char_count = len(edited_prompt)
                st.caption(f"Character count: {char_count:,}")
            
            st.divider()
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
            
            # Display tool usage statistics (only when tools have been used)
            st.divider()
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
                
        # Chat input at the bottom
        chat_input = st.chat_input("Ask me anything about SEO")

        if chat_input:
            # 🔎 Find previous assistant message content (if any), BEFORE this turn
            prev_assistant_content = None
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant":
                    prev_assistant_content = msg["content"]
                    break

            # 1) Save + show user message
            st.session_state.messages.append({"role": "user", "content": chat_input})
            with st.chat_message("user"):
                st.markdown(chat_input)

            # 2) Assistant message block with three placeholders:
            #    - spinner (top)
            #    - thinking (middle)
            #    - final response (bottom)
            with st.chat_message("assistant"):
                spinner_placeholder = st.empty()
                thinking_placeholder = st.empty()
                response_placeholder = st.empty()

                # initial spinner text
                with spinner_placeholder:
                    st.markdown("🤔 AI is thinking...")

                try:
                    # ✅ Pass prev_assistant_content into the stream function
                    response, thinking_steps = self._run_async(
                        self._stream_message(
                            chat_input,
                            thinking_placeholder,
                            response_placeholder,
                            spinner_placeholder,
                            prev_assistant_content,
                        )
                    )

                    # 3) Ensure final response text is there (if stream didn't already print)
                    if response and response_placeholder:
                        response_placeholder.markdown(response)

                    # 4) Save assistant response for future turns
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                    # 5) Optional: show full thinking steps in an expander
                    if thinking_steps:
                        with st.expander("🧠 Agent thinking details", expanded=False):
                            self._display_thinking_steps(thinking_steps)

                except Exception as e:
                    spinner_placeholder.empty()
                    err_msg = f"❌ **Error**: {str(e)}\n\nPlease try again or contact support if the issue persists."
                    response_placeholder.error(err_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err_msg}
                    )


                    
                    