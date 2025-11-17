import streamlit as st
import os
import sys
import pandas as pd # Added for the error return type

# --- FIX: Ensure the current directory is in the Python path ---
# This forces the interpreter to look in the folder where app.py resides.
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ------------------------------------------------------------------

# Now, the import should succeed if ted_recommender.py is in the same directory
from ted_recommender import TEDRecommender 

# Define the relative path to the data file
# Assuming 'ted_main.csv' is in the same directory as app.py
DATA_PATH = "ted_main.csv" 

st.set_page_config(
    page_title="TED Talks Recommendation System",
    layout="wide"
)

st.title("🎤 TED Talks Chat-Based Recommendation System")

# Use a spinner to show the user that something is happening during the initial load
with st.spinner("🚀 Loading TED Talks data and model... The first load might take a minute."):
    try:
        rec = TEDRecommender(DATA_PATH)
        st.success("System ready! Ask me anything about TED Talks.")
    except Exception as e:
        # Provide a clear error message that also prompts the user to check their files
        st.error(f"Error initializing the recommender. Please verify 'ted_recommender.py' and 'ted_main.csv' files are in the same directory as 'app.py'. Details: {e}")
        rec = None # Set rec to None to handle the condition check later

st.divider()

user_query = st.text_input(
    "💬 What topic or idea are you interested in?",
    placeholder="e.g., 'sustainable architecture', 'psychology of decision-making', 'future of AI'",
    key="query_input"
)

# Check if the recommender object 'rec' exists and is ready
# We use the explicit check against None from the error block above
if rec is None:
    st.warning("Recommendation engine is not functional due to an initialization error.")
elif st.button("Recommend Talks", use_container_width=True) or (user_query and st.session_state.query_input != st.session_state.get('last_query', '')):
    
    if user_query:
        # Store the current query to prevent re-running without change
        st.session_state.last_query = user_query 
        
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