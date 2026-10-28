
import streamlit as st
from chatbot import generate_response

st.set_page_config(
    page_title="InnerGlow | Agentic AI",
    page_icon="💚",
    layout="centered"
)

st.title("💚 InnerGlow")
st.caption("An Agentic AI Mental Wellbeing Chatbot for University Students")

with st.expander("🤖 How InnerGlow Works"):
    st.write("""
    InnerGlow uses multiple AI agents and agentic design patterns:

    - **Emotion Support Agent:** Generates supportive responses.
    - **CBT Knowledge Agent:** Retrieves relevant knowledge using RAG.
    - **Web Search Agent:** Searches for external information.
    - **Routing:** Selects the appropriate knowledge source.
    - **Reflection:** Reviews and improves generated responses.

    Powered by Groq LLMs and ChromaDB.
    """)

# Store conversation separately for each Streamlit session.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear conversation button.
if st.button("🗑️ Clear conversation"):
    st.session_state.messages = []
    st.rerun()

# Display existing conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("How are you feeling today?")

if user_input:

    # Build conversation history BEFORE adding the new message.
    previous_messages = st.session_state.messages.copy()

    history = []

    for i in range(0, len(previous_messages) - 1, 2):
        user_turn = previous_messages[i]
        assistant_turn = previous_messages[i + 1]

        if (
            user_turn["role"] == "user"
            and assistant_turn["role"] == "assistant"
        ):
            history.append({
                "user": user_turn["content"],
                "assistant": assistant_turn["content"]
            })

    # Keep the most recent five conversation turns.
    history = history[-5:]

    # Display and store the new user message.
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate the assistant response using conversation history.
    with st.chat_message("assistant"):
        with st.spinner("InnerGlow is thinking..."):
            try:
                reply = generate_response(
                    message=user_input,
                    emotion="neutral",
                    history=history
                )

                st.markdown(reply)

            except Exception:
                reply = (
                    "Sorry, I'm having trouble responding right now. "
                    "Please try again shortly."
                )
                st.error(reply)

    # Store the assistant response for future turns.
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })
