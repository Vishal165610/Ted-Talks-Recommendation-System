import streamlit as st
import os
import sys
import pandas as pd

# ---------------------------------------------------------
# PATH FIXES
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(current_dir, ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from ted_recommender import TEDRecommender
except ImportError:
    st.error("Error: Cannot find 'ted_recommender.py'. Check your folder structure.")
    st.stop()

DATA_PATH = os.path.join(BASE_DIR, "data", "ted_main.csv")
# ---------------------------------------------------------


# ---------------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="TED Talks Recommendation System",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
if 'rec' not in st.session_state:
    with st.spinner("🚀 Loading TED Talks dataset & model..."):
        try:
            st.session_state.rec = TEDRecommender(DATA_PATH)
            st.toast("System ready!")
        except Exception as e:
            st.error(f"Failed to initialize recommender: {e}")
            st.session_state.rec = None

if st.session_state.rec is None:
    st.stop()


# ---------------------------------------------------------
# CHAT MEMORY
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! Ask me about any topic and I’ll recommend TED talks."}
    ]


# ---------------------------------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------------------------------
st.title("🎤 TED Talks Chat-Based Recommender")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):

        # If assistant sent recommendations
        if msg["role"] == "assistant" and "results" in msg:
            st.markdown(msg["content"])
            st.subheader("🔎 Top Recommended TED Talks")

            for index, row in msg["results"].iterrows():
                st.markdown(
                    f"""
                    <div style="padding: 15px; border-radius: 12px; margin-bottom: 15px;
                        background-color: #F8F9FA; border-left: 5px solid #E62B1F;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                        <h5 style="margin-top: 0; color: #1E2B38; font-weight: 600;">
                            {row['title']}
                        </h5>
                        <p style="margin-bottom: 0;">
                            🗣️ <strong>Speaker:</strong> {row['main_speaker']} <br>
                            ⭐ <strong>Views:</strong> {row['views']:,} <br>
                            🔗 <a href="{row['url']}" target="_blank"
                                style="color: #E62B1F; text-decoration: none;">
                                Watch Talk
                            </a>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown(msg["content"])


# ---------------------------------------------------------
# SINGLE CHAT INPUT (IMPORTANT — only one)
# ---------------------------------------------------------
user_query = st.chat_input("Ask about a topic or idea...")

if user_query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    # Process request
    with st.chat_message("assistant"):
        with st.spinner(f"Searching for TED talks about '{user_query}'..."):
            try:
                results_df = st.session_state.rec.recommend(user_query)

                # Save assistant message with results
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Here are the top TED Talks related to **{user_query}**:",
                    "results": results_df
                })

                # Show results immediately
                st.markdown(f"Here are the top TED Talks related to **{user_query}**:")
                st.subheader("🔎 Top Recommended TED Talks")

                for index, row in results_df.iterrows():
                    st.markdown(
                        f"""
                        <div style="padding: 15px; border-radius: 12px; margin-bottom: 15px;
                            background-color: #F8F9FA; border-left: 5px solid #E62B1F;
                            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                            <h5 style="margin-top: 0; color: #1E2B38; font-weight: 600;">
                                {row['title']}
                            </h5>
                            <p>
                                🗣️ <strong>Speaker:</strong> {row['main_speaker']} <br>
                                ⭐ <strong>Views:</strong> {row['views']:,} <br>
                                🔗 <a href="{row['url']}" target="_blank" style="color: #E62B1F;">
                                    Watch Talk
                                </a>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error while recommending: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, I couldn't process your request: {e}"
                })

    st.rerun()
