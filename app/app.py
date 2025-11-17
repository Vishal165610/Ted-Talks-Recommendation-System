import streamlit as st
import pandas as pd
import sys
import os

# --------------------------------------
# FIX IMPORT PATH FOR STREAMLIT CLOUD
# --------------------------------------
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from src.ted_recommender import TEDRecommender

# --------------------------------------
# LOAD MODEL + DATA
# --------------------------------------
DATA_PATH = os.path.join(parent_dir, "data", "ted_talks.csv")

st.title("🎤 TED Talks Recommendation System")

rec = TEDRecommender(DATA_PATH)

# --------------------------------------
# UI DROPDOWN
# --------------------------------------
title_list = rec.df['title'].tolist()

selected_title = st.selectbox("Select a TED Talk:", title_list)

# --------------------------------------
# SHOW RECOMMENDATIONS
# --------------------------------------
if st.button("Recommend"):
    results = rec.recommend(selected_title)
    
    st.subheader("🔎 Recommended Talks:")
    st.table(results)
