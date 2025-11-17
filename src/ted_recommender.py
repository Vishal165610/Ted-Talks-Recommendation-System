import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TEDRecommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # combine text fields to create strong embeddings
        self.df['combined'] = (
            self.df['title'].fillna('') + " " +
            self.df['main_speaker'].fillna('') + " " +
            self.df['description'].fillna('') + " " +
            self.df['tags'].astype(str)
        )

        # load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # prepare embeddings for each talk
        self.embeddings = self.model.encode(
            self.df['combined'].tolist(),
            show_progress_bar=True
        )

    def recommend(self, user_query, top_n=5):

        # embed user input
        query_emb = self.model.encode([user_query])

        # compute cosine similarity
        scores = cosine_similarity(query_emb, self.embeddings)[0]

        # get top N matches
        top_idx = scores.argsort()[-top_n:][::-1]

        # return selected useful columns
        return self.df.iloc[top_idx][['title', 'main_speaker', 'url']]
