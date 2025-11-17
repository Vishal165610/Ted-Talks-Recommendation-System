import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TEDRecommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['combined'] = self.df['title'] + " " + self.df['description']

        # Load model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Create embeddings for all talks
        self.embeddings = self.model.encode(self.df['combined'], show_progress_bar=True)

    def recommend(self, user_query, top_n=5):
        # Embed the user's question/title
        query_embedding = self.model.encode([user_query])

        # Calculate cosine similarity
        scores = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top N results
        top_indices = scores.argsort()[-top_n:][::-1]

        # Return titles + speakers
        return self.df.iloc[top_indices][['title', 'speaker']]
