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
    Loads the SBERT model and computes/loads talk embeddings, cached by Streamlit.
    This function handles the slow part of the process, running only once.
    """
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        # Catch generic exceptions during read for robustness
        st.error(f"❌ Error loading data from CSV file: {e}")
        st.info("Please ensure 'ted_main.csv' is in the expected location.")
        return None, None, None
    
    # 1. Combine text fields to create strong embeddings
    df['combined'] = (
        df['title'].fillna('') + " " +
        df['main_speaker'].fillna('') + " " +
        df['description'].fillna('') + " " +
        df['tags'].astype(str)
    )

    # 2. Load the embedding model (cached)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Load or compute embeddings
    if os.path.exists(EMBEDDINGS_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
    else:
        st.info("⏳ Computing embeddings for the first time... this may take a moment.")
        embeddings = model.encode(
            df['combined'].tolist(),
            show_progress_bar=False 
        )
        np.save(EMBEDDINGS_FILE, embeddings)
        st.success(f"✅ Embeddings computed and saved to {EMBEDDINGS_FILE}")
        
    return df, model, embeddings

class TEDRecommender:
    def __init__(self, csv_path):
        # Load everything using the cached function
        self.df, self.model, self.embeddings = load_model_and_embeddings(csv_path)

        # Pre-calculate normalized popularity scores
        self.normalized_views = None
        if self.df is not None:
            if 'views' not in self.df.columns:
                 # Check for required 'views' column
                 st.error("Missing 'views' column in CSV. Cannot calculate popularity scores.")
                 return

            # Use log1p (log(1 + x)) to handle the skewed view counts
            view_scores = np.log1p(self.df['views'].values)
            
            # ZeroDivisionError Fix: Check if range is zero
            score_range = view_scores.max() - view_scores.min()
            
            if score_range == 0:
                # If all views are the same, assign a neutral score (0.5)
                self.normalized_views = np.full(view_scores.shape, 0.5) 
            else:
                # Min-Max Scaling to put views in the 0-1 range
                self.normalized_views = (view_scores - view_scores.min()) / score_range
            
        else:
            self.normalized_views = None

    def recommend(self, user_query, top_n=5, alpha=0.6):
        """
        Recommends talks by blending semantic similarity (relevance) and popularity (views).
        
        Args:
            user_query (str): The user's input query.
            top_n (int): The number of top talks to return.
            alpha (float): Blending factor (0.0 to 1.0). Higher alpha favors RELEVANCE.
        """
        if self.df is None or self.normalized_views is None:
            # Return empty DataFrame or error if initialization failed
            return pd.DataFrame({'title': ['Error: Recommender not ready'], 'main_speaker': ['N/A'], 'url': ['#'], 'views': [0]})

        # 1. Embed user input
        query_emb = self.model.encode([user_query])

        # 2. Compute cosine similarity (Relevance Score)
        scores = cosine_similarity(query_emb, self.embeddings)[0]

        # 3. Combine scores for "Smartness": total_score = (alpha * relevance) + ((1 - alpha) * popularity)
        total_scores = (alpha * scores) + ((1 - alpha) * self.normalized_views)

        # 4. Get top N matches based on the combined score
        top_idx = total_scores.argsort()[-top_n:][::-1]

        # 5. Return selected useful columns
        return self.df.iloc[top_idx][['title', 'main_speaker', 'url', 'views']]