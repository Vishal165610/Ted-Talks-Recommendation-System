import streamlit as st
import os
import sys

# ---------------------------------------------------------
# FIX: Add `/src` folder so we can import ted_recommender
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from ted_recommender import TEDRecommender
# ---------------------------------------------------------

DATA_PATH = os.path.join(BASE_DIR, "data", "ted_main.csv")


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="TEDX Chat Recommender",
    layout="wide",
    page_icon="🎤"
)

# ---------------------------------------------------------
# Custom CSS for ChatGPT-like UI
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        body { background-color: #f6f6f6; }

        .chat-container {
            max-width: 850px;
            margin-left: auto;
            margin-right: auto;
            padding: 20px;
        }

        .user-message {
            background-color: #DCF8C6;
            padding: 12px 18px;
            border-radius: 12px;
            margin-bottom: 12px;
            max-width: 70%;
            margin-left: auto;
            color: #111;
            font-size: 1.1rem;
        }

        .bot-message {
            background-color: #FFFFFF;
            padding: 12px 18px;
            border-radius: 12px;
            margin-bottom: 12px;
            max-width: 70%;
            margin-right: auto;
            border: 1px solid #ddd;
            font-size: 1.1rem;
        }

        .recommend-box {
            background: #fafafa;
            border-left: 4px solid #e62b1f;
            padding: 12px 15px;
            margin: 12px 0;
            border-radius: 8px;
        }

        .chat-title {
            text-align: center;
            font-size: 2rem;
            font-weight: bold;
            padding-top: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Load the recommender
# ---------------------------------------------------------
with st.spinner("Loading TED Talks model..."):
    try:
        rec = TEDRecommender(DATA_PATH)
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        st.stop()

# ---------------------------------------------------------
# Chat Session State
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! 👋 Ask me about any topic, and I'll recommend the best TED Talks."}
    ]


# ---------------------------------------------------------
# Chat Display
# ---------------------------------------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">{msg["content"]}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# User Input
# ---------------------------------------------------------
user_query = st.chat_input("Ask about any topic...")

if user_query:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Get recommendations
    results = rec.recommend(user_query)

    # Build response message
    bot_reply = "Here are the top TED Talk recommendations for your interest:\n\n"

    for i, row in results.iterrows():
        bot_reply += (
            f"""
            <div class='recommend-box'>
                <strong>{row['title']}</strong> <br>
                👤 {row['main_speaker']} <br>
                👀 {row['views']:,} views <br>
                🔗 <a href="{row['url']}" target="_blank">Watch Talk</a>
            </div>
            """
        )

    # Add bot reply
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    # Rerun so messages update instantly
    st.rerun()
