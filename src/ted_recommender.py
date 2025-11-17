import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import streamlit as st 
import math

# --- Configuration ---
EMBEDDINGS_FILE = "ted_talks_embeddings.npy"
# ---

@st.cache_resource
def load_model_and_embeddings(csv_path):
    """
    Loads the SBERT model and computes/loads talk embeddings, cached by Streamlit.
    """
    
    try:
        # Catch a generic Exception here to prevent 'df' from being accessed
        # if reading fails for any reason (like path, decoding, etc.)
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"❌ Error loading data from CSV file: {e}")
        st.info("Please ensure 'ted_main.csv' is in the root directory or adjust DATA_PATH in app.py.")
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
        if self.df is not None:
            # Check for required 'views' column
            if 'views' not in self.df.columns:
                 st.error("Missing 'views' column in CSV. Cannot calculate popularity scores.")
                 self.normalized_views = None
                 return

            # Use log1p (log(1 + x)) to handle the skewed view counts
            view_scores = np.log1p(self.df['views'].values)
            
            # --- FIX FOR ZeroDivisionError ---
            score_range = view_scores.max() - view_scores.min()
            
            if score_range == 0:
                # If all views are the same, assign a neutral score (0.5) to all talks
                self.normalized_views = np.full(view_scores.shape, 0.5) 
            else:
                # Min-Max Scaling to put views in the 0-1 range
                self.normalized_views = (view_scores - view_scores.min()) / score_range
            # --- END FIX ---
            
        else:
            self.normalized_views = None

    def recommend(self, user_query, top_n=5, alpha=0.6):
        """
        Recommends talks by blending semantic similarity (relevance) and popularity (views).
        
        Alpha controls the weight of relevance (0.0 to 1.0).
        """
        if self.df is None or self.normalized_views is None:
            return pd.DataFrame({'title': ['Error: Recommender not fully initialized.'], 'main_speaker': ['N/A'], 'url': ['#'], 'views': [0]})

        # 1. Embed user input
        query_emb = self.model.encode([user_query])

        # 2. Compute cosine similarity (Relevance Score)
        scores = cosine_similarity(query_emb, self.embeddings)[0]

        # 3. Combine scores: total_score = (alpha * relevance) + ((1 - alpha) * popularity)
        # This blends the semantic match with the talk's general popularity (views).
        total_scores = (alpha * scores) + ((1 - alpha) * self.normalized_views)

        # 4. Get top N matches based on the combined score
        top_idx = total_scores.argsort()[-top_n:][::-1]

        # 5. Return selected useful columns
        return self.df.iloc[top_idx][['title', 'main_speaker', 'url', 'views']]