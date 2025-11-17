import streamlit as st
import sys
import os

# Fix import path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from src.ted_recommender import TEDRecommender

DATA_PATH = os.path.join(parent_dir, "data", "ted_main.csv")

st.title("🎤 TED Talks Chat-Based Recommendation System")

rec = TEDRecommender(DATA_PATH)

user_query = st.text_input("Ask about a topic, title, or idea:")

if st.button("Recommend"):
    results = rec.recommend(user_query)

    st.subheader("🔎 Recommended TED Talks")
    st.write(results)
