import streamlit as st
import os

# Assuming ted_recommender.py is in a 'src' folder at the same level as 'app.py'
# If it's not, you may need a minimal __init__.py in the 'src' folder
from ted_recommender import TEDRecommender 

# Define the relative path to the data file
# Adjust this based on your actual file location relative to app.py
DATA_PATH = "ted_main.csv" 

st.set_page_config(
    page_title="TED Talks Recommendation System",
    layout="wide"
)

st.title("🎤 TED Talks Chat-Based Recommendation System")

# Use a spinner to show the user that something is happening during the initial load
# This addresses Challenge 2 (delayed input) by providing feedback.
with st.spinner("🚀 Loading TED Talks data and model... The first load might take a minute."):
    try:
        rec = TEDRecommender(DATA_PATH)
        st.success("System ready! Ask me anything about TED Talks.")
    except Exception as e:
        st.error(f"Error initializing the recommender: {e}")
        st.stop() # Stop execution if initialization fails

st.divider()

user_query = st.text_input(
    "💬 What topic or idea are you interested in?",
    placeholder="e.g., 'sustainable architecture', 'psychology of decision-making', 'future of AI'",
    key="query_input"
)

if st.button("Recommend Talks", use_container_width=True) or user_query:
    if user_query:
        with st.spinner("Searching for the best talks..."):
            results_df = rec.recommend(user_query)

        st.subheader("🔎 Top Recommended TED Talks")
        
        # Display results in a better format
        for index, row in results_df.iterrows():
            st.markdown(
                f"""
                <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: #F0F2F6;">
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
    else:
        st.warning("Please enter a query to get recommendations.")