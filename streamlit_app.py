
import streamlit as st
from chatbot import generate_response

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="InnerGlow | Your Wellbeing Companion",
    page_icon="💚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(
            135deg,
            #f8f7ff 0%,
            #eefaf7 50%,
            #fdfaff 100%
        );
        color: #253247;
    }

    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

    h1, h2, h3 {
        color: #29364d;
    }

    .hero {
        text-align: center;
        padding: 1.8rem 1rem;
        margin-bottom: 1.5rem;
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid #e6e6f4;
        border-radius: 24px;
        box-shadow: 0 8px 30px rgba(75, 75, 110, 0.05);
    }

    .hero-icon {
        font-size: 2.7rem;
        margin-bottom: 0.4rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #5261a6;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #65718a;
        line-height: 1.7;
    }

    .welcome-card {
        background: #ffffff;
        padding: 1.3rem 1.5rem;
        border-radius: 18px;
        border: 1px solid #e7eaf4;
        margin-bottom: 1.3rem;
        color: #334155;
    }

    .welcome-title {
        font-weight: 700;
        font-size: 1.15rem;
        color: #45538e;
        margin-bottom: 0.5rem;
    }

    [data-testid="stChatMessage"] {
        border-radius: 18px;
        padding: 0.8rem 1rem;
        border: 1px solid #e8eaf1;
        background-color: #ffffff;
        margin-bottom: 0.7rem;
    }

    [data-testid="stChatInput"] {
        border-radius: 18px;
    }

    [data-testid="stSidebar"] {
        background-color: #f2f4fc;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #45538e;
    }

    .disclaimer {
        font-size: 0.8rem;
        color: #64748b;
        text-align: center;
        padding: 1rem 0;
        line-height: 1.6;
    }

    div.stButton > button {
        border-radius: 12px;
        font-weight: 600;
    }

    @media (prefers-color-scheme: dark) {
        .stApp {
            background: #151923;
            color: #e5e7eb;
        }

        .hero, .welcome-card,
        [data-testid="stChatMessage"] {
            background: #222938;
            border-color: #374151;
            color: #e5e7eb;
        }

        .hero-title, .welcome-title,
        h1, h2, h3 {
            color: #b9c4ff;
        }

        .hero-subtitle, .disclaimer {
            color: #a9b4c7;
        }

        [data-testid="stSidebar"] {
            background-color: #1d2432;
        }
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "suggested_prompt" not in st.session_state:
    st.session_state.suggested_prompt = None

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.title("💚 InnerGlow")
    st.caption("A little support, whenever you need it.")

    st.divider()

    st.markdown("### 🌿 Your space")
    st.write(
        "A supportive AI companion for university "
        "students navigating everyday challenges."
    )

    if st.button(
        "🗑️ Start a new conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.suggested_prompt = None
        st.rerun()

    st.divider()

    with st.expander("🤖 How InnerGlow works"):
        st.markdown("""
        InnerGlow combines:

        - **Emotion Support Agent**
        - **CBT Knowledge Agent**
        - **Web Search Agent**
        - **Intelligent Routing**
        - **Reflection and Self-Review**

        Powered by Groq LLMs and ChromaDB RAG.
        """)

    with st.expander("🔒 Privacy and wellbeing"):
        st.markdown("""
        - Conversations are kept in your current
          Streamlit session.
        - Avoid sharing sensitive personal details.
        - AI responses are for general wellbeing
          support, not professional diagnosis.
        - Messages are processed by external
          AI services to generate responses.
        """)

    st.divider()
    st.caption("Made with care for university students 💚")

# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-icon">🌱</div>
    <div class="hero-title">InnerGlow</div>
    <div class="hero-subtitle">
        Your supportive space to talk, reflect, and grow.
        <br>
        Whatever is on your mind, you can start here.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# WELCOME SCREEN
# --------------------------------------------------
if not st.session_state.messages:

    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-title">
            💬 Hey there! How are you feeling today?
        </div>
        You don't need to have everything figured out.
        Whether you're stressed about studies, feeling
        overwhelmed, or simply want someone to talk to,
        InnerGlow is here to listen.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### ✨ Not sure where to start?")

    prompts = [
        ("📚 Exam stress", "I'm feeling stressed about my exams."),
        ("🌧️ Feeling low", "I've been feeling a little down lately."),
        ("🌿 Need calm", "Can you help me relax and feel calmer?"),
        ("💬 Let's talk", "Hi InnerGlow, I just want to talk.")
    ]

    col1, col2 = st.columns(2)

    for index, (label, prompt) in enumerate(prompts):
        column = col1 if index % 2 == 0 else col2

        with column:
            if st.button(
                label,
                key=f"starter_{index}",
                use_container_width=True
            ):
                st.session_state.suggested_prompt = prompt
                st.rerun()

# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------
for message in st.session_state.messages:

    avatar = "🌱" if message["role"] == "assistant" else "🙂"

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):
        st.markdown(message["content"])

# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------
typed_input = st.chat_input(
    "Share what's on your mind..."
)

user_input = typed_input

if st.session_state.suggested_prompt is not None:
    user_input = st.session_state.suggested_prompt
    st.session_state.suggested_prompt = None

# --------------------------------------------------
# CHATBOT PROCESSING
# --------------------------------------------------
if user_input and user_input.strip():

    # Preserve existing history logic
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

    history = history[-5:]

    # Save and display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user", avatar="🙂"):
        st.markdown(user_input)

    # Generate chatbot response
    with st.chat_message("assistant", avatar="🌱"):

        with st.spinner("InnerGlow is here with you..."):

            try:
                reply = generate_response(
                    message=user_input,
                    emotion="neutral",
                    history=history
                )

                st.markdown(reply)

            except Exception:
                reply = (
                    "Sorry, I'm having trouble responding "
                    "right now. Please try again shortly."
                )
                st.error(reply)

    # Store response for conversation memory
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="disclaimer">
    💚 InnerGlow is an AI wellbeing companion,
    not a replacement for professional mental
    health support.
    <br>
    If you are in immediate danger,
    contact local emergency services.
</div>
""", unsafe_allow_html=True)
