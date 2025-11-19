import streamlit as st
import asyncio
import json
from src.agents.seo_agent import SEOAgent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


class AppUI:    
    def __init__(self):
        self.title = "🔍 SEO MCP Agent Chat"
        self.description = "Chat with your SEO expert agent powered by DataForSEO and Google Search Console"
        
        # Initialize agent in session state if not already initialized
        if "seo_agent" not in st.session_state:
            model_name = st.session_state.get("selected_model", "gpt-4o-mini")
            llm = ChatOpenAI(model=model_name, temperature=0, timeout=120*60) # 2 hrs
            st.session_state.seo_agent = SEOAgent(llm=llm)
        
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
    
    async def _stream_message(self, user_message: str, thinking_placeholder, response_placeholder):
        """Stream agent responses and update the UI in real-time with thinking process."""
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
        
        # Return the final response and thinking steps
        return full_response if full_response else "I apologize, but I couldn't generate a response. Please try again.", thinking_steps
    
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
                st.session_state.selected_model = selected_model
                llm = ChatOpenAI(model=selected_model, temperature=0)
                st.session_state.seo_agent = SEOAgent(llm=llm)
            
            st.divider()
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()

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
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})