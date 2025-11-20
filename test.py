import time
import streamlit as st

st.set_page_config(page_title="Chat Demo", page_icon="💬")

st.title("💬 Simple Chat Demo with chat_message")

# Initialize message history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm a demo bot. Ask me anything 👋"}
    ]


def bot_reply(user_message: str) -> str:
    """
    Predefined agent reply.
    You can make this as fancy as you like later.
    """
    return (
        "🤖 *Predefined reply:*\n\n"
        f"I heard you say:\n> `{user_message}`\n\n"
        "Right now I'm using a fixed response with a 2-second thinking delay."
    )


# 1) Render the existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2) Chat input at the bottom
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Agent "thinking" + response
    with st.chat_message("assistant"):
        
        with st.spinner("Thinking..."):
            time.sleep(2)  # 2-second delay to simulate thinking
            response = bot_reply(user_input)
            st.markdown(response)

    # Save agent message to history
    st.session_state.messages.append({"role": "assistant", "content": response})
