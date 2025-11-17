
import streamlit as st
import pandas as pd
from ted_recommender import TEDRecommender

st.title("TED Talks Recommendation System")

rec = TEDRecommender("data/ted_talks.csv")

title_list = rec.df['title'].tolist()
selected_title = st.selectbox("Select a TED Talk:", title_list)

if st.button("Recommend"):
    results = rec.recommend(selected_title)
    st.write("### Recommended Talks:")
    st.table(results)
