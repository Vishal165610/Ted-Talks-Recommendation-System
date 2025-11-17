import streamlit as st
import os
import sys

# --- FIX: Ensure Python can find ted_recommender.py (ModuleNotFoundError) ---
# Add the directory containing app.py and potentially its parent to the Python path
# This handles the case where ted_recommender.py is in the same directory, 
# or a 'src' folder sibling to 'app.py's parent.
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Add current and parent directory to path to find the module
if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Now, the import should succeed if the file is accessible in these paths.
from ted_recommender import TEDRecommender 
# --------------------------------------------------------------------------

# Define the relative path to the data file
# Assuming 'ted_main.csv' is in the same directory as app.py for simplicity
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
        # Pass the path to the recommender
        rec = TEDRecommender(DATA_PATH)
        st.success("System ready! Ask me anything about TED Talks.")
    except Exception as e:
        # The recommender's error handling will print a more specific message, 
        # but this catches any remaining issues during initialization.
        st.error(f"Error initializing the recommender. Please check your file paths and dependencies: {e}")
        # st.stop() # Removed st.stop() to allow the rest of the UI to load, even if the recommender fails
        # This provides a better user experience by allowing them to see the error and input field.

st.divider()

user_query = st.text_input(
    "💬 What topic or idea are you interested in?",
    placeholder="e.g., 'sustainable architecture', 'psychology of decision-making', 'future of AI'",
    key="query_input"
)

# Use st.session_state to track if the recommendation engine is ready
if 'rec' not in locals() and 'rec' not in globals():
    st.warning("Recommendation engine is not ready. Please resolve the initialization error above.")
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