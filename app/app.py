import streamlit as st
import sys
import os
import pandas as pd

# Fix path for Streamlit Cloud
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from src.ted_recommender import TEDRecommender

# Path to dataset
DATA_PATH = os.path.join(parent_dir, "data", "ted_talks.csv")

# Load model
rec = TEDRecommender(DATA_PATH)

st.title("🎤 TED Talks Chat Recommendation System")

st.write("Ask me anything related to TED Talks. Enter a title or describe a topic.")

# Chat input box
user_input = st.text_input("Your Query (e.g., happiness, AI, motivation, etc.):")

if st.button("Get Recommendations"):
    if user_input.strip():
        results = rec.recommend(user_input)
        st.subheader("🔎 Recommended Talks:")
        st.table(results)
    else:
        st.warning("Please enter a valid query!")
