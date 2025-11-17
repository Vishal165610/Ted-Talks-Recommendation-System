import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import streamlit as st # Necessary for the caching decorator
import math

# --- Configuration ---
EMBEDDINGS_FILE = "ted_talks_embeddings.npy"
# ---

@st.cache_resource
def load_model_and_embeddings(csv_path):
    """
    Loads the SBERT model and computes/loads talk embeddings.
    
    This function is cached by Streamlit (@st.cache_resource) 
    and will only run once, solving Challenges 1 and 2 (slow loading).
    """
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Error: CSV file not found at path '{csv_path}'. Please ensure 'ted_main.csv' is in the correct location.")
        return None, None, None

    # 1. Combine text fields to create strong embeddings
    df['combined'] = (
        df['title'].fillna('') + " " +
        df['main_speaker'].fillna('') + " " +
        df['description'].fillna('') + " " +
        df['tags'].astype(str)
    )

    # 2. Load the embedding model (cached)
    # Using 'all-MiniLM-L6-v2' as specified
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Load or compute embeddings
    if os.path.exists(EMBEDDINGS_FILE):
        # Load from file if they exist
        embeddings = np.load(EMBEDDINGS_FILE)
        # st.success("Loaded pre-computed embeddings!") # Commented out for cleaner UI
    else:
        # Compute and save if they don't exist (this is the slow step, runs once)
        st.info("Computing embeddings for the first time... this may take a moment.")
        embeddings = model.encode(
            df['combined'].tolist(),
            show_progress_bar=False 
        )
        np.save(EMBEDDINGS_FILE, embeddings)
        st.success(f"Embeddings computed and saved to {EMBEDDINGS_FILE}")
        
    return df, model, embeddings

class TEDRecommender:
    def __init__(self, csv_path):
        # Load everything using the cached function
        self.df, self.model, self.embeddings = load_model_and_embeddings(csv_path)

        # Pre-calculate normalized popularity scores
        if self.df is not None:
            # Use log1p (log(1 + x)) to handle the skewed view counts
            view_scores = np.log1p(self.df['views'].values) 
            # Min-Max Scaling to put views in the 0-1 range
            self.normalized_views = (view_scores - view_scores.min()) / (view_scores.max() - view_scores.min())
        else:
            self.normalized_views = None

    def recommend(self, user_query, top_n=5, alpha=0.6):
        """
        Recommends talks by blending semantic similarity (relevance) and popularity (views).
        
        Args:
            user_query (str): The user's input query.
            top_n (int): The number of top talks to return.
            alpha (float): The blending factor (0.0 to 1.0). 
                           Higher alpha means more weight on RELEVANCE.
                           (e.g., 0.6 means 60% relevance, 40% popularity).
        """
        if self.df is None:
            return pd.DataFrame()

        # 1. Embed user input
        query_emb = self.model.encode([user_query])

        # 2. Compute cosine similarity (Relevance Score)
        scores = cosine_similarity(query_emb, self.embeddings)[0]

        # --- Improvement for "Smartness" (Challenge 3: Relevance/Popularity Blend) ---
        
        # 3. Combine scores: total_score = (alpha * relevance) + ((1 - alpha) * popularity)
        # We use the pre-calculated self.normalized_views
        total_scores = (alpha * scores) + ((1 - alpha) * self.normalized_views)

        # 4. Get top N matches based on the combined score
        top_idx = total_scores.argsort()[-top_n:][::-1]

        # 5. Return selected useful columns
        return self.df.iloc[top_idx][['title', 'main_speaker', 'url', 'views']]