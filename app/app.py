import streamlit as st
import os
import sys
import pandas as pd

# ---------------------------------------------------------
# PATH FIXES (Based on your provided structure)
# ---------------------------------------------------------
# Assuming a structure where app.py is in a subdirectory (e.g., 'app')
# and 'src', 'data' are in the parent directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(current_dir, ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Import the recommender model
try:
    from ted_recommender import TEDRecommender
except ImportError:
    st.error("Error: Cannot find 'ted_recommender.py'. Please check your file structure and path settings.")
    st.stop()
    
# Correct path to CSV inside /data folder
DATA_PATH = os.path.join(BASE_DIR, "data", "ted_main.csv")
# ---------------------------------------------------------


# --- Setup & Initialization ---

# Streamlit Page Config
st.set_page_config(
    page_title="TED Talks Recommendation System",
    layout="wide", # Use wide layout for a cleaner look
    initial_sidebar_state="collapsed" # Hide sidebar for focus
)

# Initialize the recommender only once using session state
if 'rec' not in st.session_state:
    with st.spinner("🚀 Loading TED Talks data & model... This may take a moment..."):
        try:
            st.session_state.rec = TEDRecommender(DATA_PATH)
            st.toast("System ready! Start chatting.")
        except Exception as e:
            st.error(f"Error initializing the recommender. Please check file paths and dependencies. Details: {e}")
            st.session_state.rec = None # Set to None on failure

if st.session_state.rec is None:
    st.title("System Error")
    st.error("Recommendation system failed to load. Please fix the error above.")
    st.stop()


# Initialize chat history (for GPT-like persistence)
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a friendly welcome message
    st.session_state.messages.append({"role": "assistant", "content": "Welcome to the TED Talks Recommender! Ask me about any topic, title, or idea, and I'll recommend the best talks."})


# --- Main Chat Interface ---

st.title("🎤 TED Talks Chat-Based Recommender")

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Use Markdown for message content display
        if message["role"] == "assistant" and "results" in message:
            # Display results in the nice, stylized format
            st.subheader("🔎 Top Recommended TED Talks")
            for index, row in message["results"].iterrows():
                st.markdown(
                    f"""
                    <div style="padding: 15px; border-radius: 12px; margin-bottom: 15px; background-color: #F8F9FA; border-left: 5px solid #E62B1F; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                        <h5 style="margin-top: 0; color: #1E2B38; font-weight: 600;">{row['title']}</h5>
                        <p style="margin-bottom: 0;">
                            🗣️ <strong>Speaker:</strong> {row['main_speaker']} <br>
                            ⭐ <strong>Views:</strong> {row['views']:,} <br>
                            🔗 <a href="{row['url']}" target="_blank" style="color: #E62B1F; text-decoration: none;">Watch Talk</a>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown(message["content"])

# User Input (Always at the bottom)
if user_query := st.chat_input("Ask about a topic or idea..."):
    # 1. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # 2. Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    # 3. Get recommendation and display assistant response
    with st.chat_message("assistant"):
        with st.spinner(f"Searching for talks related to '{user_query}'..."):
            try:
                results_df = st.session_state.rec.recommend(user_query)

                # Store the results DataFrame in the message history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Here are the top TED Talks I found matching your query: **{user_query}**",
                    "results": results_df # Store the DataFrame for display
                })

                # Manually display the results for the current turn (will be repeated by the loop on next run)
                st.markdown(f"Here are the top TED Talks I found matching your query: **{user_query}**")
                st.subheader("🔎 Top Recommended TED Talks")

                for index, row in results_df.iterrows():
                    st.markdown(
                        f"""
                        <div style="padding: 15px; border-radius: 12px; margin-bottom: 15px; background-color: #F8F9FA; border-left: 5px solid #E62B1F; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                            <h5 style="margin-top: 0; color: #1E2B38; font-weight: 600;">{row['title']}</h5>
                            <p style="margin-bottom: 0;">
                                🗣️ <strong>Speaker:</strong> {row['main_speaker']} <br>
                                ⭐ <strong>Views:</strong> {row['views']:,} <br>
                                🔗 <a href="{row['url']}" target="_blank" style="color: #E62B1F; text-decoration: none;">Watch Talk</a>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            except Exception as e:
                error_message = f"I'm sorry, I ran into an error while generating recommendations: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                
    # Rerun the script to update the chat history display
if user_query := st.chat_input("Ask about a topic or idea..."):

    # processing messages...

    # Rerun the script to update the chat history display
    st.rerun()   # ← inside the block
