import streamlit as st
import os
import sys
import pandas as pd

# ---------------------------------------------------------
# FIX: Add `/src` folder to Python path so imports work
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Now import the recommender model from /src
from ted_recommender import TEDRecommender
# ---------------------------------------------------------


# ---------------------------------------------------------
# FIX: Correct path to CSV inside /data folder
# ---------------------------------------------------------
DATA_PATH = os.path.join(BASE_DIR, "data", "ted_main.csv")
# ---------------------------------------------------------


# Streamlit Page Config
st.set_page_config(
    page_title="TED Talks Recommendation System",
    layout="wide"
)

st.title("🎤 TED Talks Chat-Based Recommendation System")

# Show spinner during initialization
with st.spinner("🚀 Loading TED Talks data & model... This may take a moment..."):
    try:
        rec = TEDRecommender(DATA_PATH)
        st.success("System ready! Ask me anything about TED Talks.")
    except Exception as e:
        st.error(f"Error initializing the recommender: {e}")
        st.stop()

st.divider()

# User Input
user_query = st.text_input(
    "💬 What topic or idea are you interested in?",
    placeholder="e.g., 'sustainable architecture', 'psychology of decision-making', 'future of AI'",
    key="query_input"
)

# Recommend Button Logic
if st.button("Recommend Talks", use_container_width=True):

    if not user_query.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Finding the best TED Talks for you..."):
            results_df = rec.recommend(user_query)

        st.subheader("🔎 Top Recommended TED Talks")

        for index, row in results_df.iterrows():
            st.markdown(
                f"""
                <div style="padding: 15px; border-radius: 8px; margin-bottom: 15px; background-color: #F0F2F6;">
                    <h4 style="margin-top: 0; color: #E62B1F;">{row['title']}</h4>
                    <p>
                        <strong>Speaker:</strong> {row['main_speaker']} <br>
                        <strong>Views:</strong> {row['views']:,} <br>
                        🔗 <a href="{row['url']}" target="_blank">Watch Talk</a>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
