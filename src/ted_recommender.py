
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TEDRecommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.df['combined'] = self.df['title'] + ' ' + self.df['description']
        self.embeddings = self.model.encode(self.df['combined'].tolist(), show_progress_bar=True)

    def recommend(self, title, top_n=5):
        if title not in self.df['title'].values:
            return ["Title not found in dataset."]
        idx = self.df[self.df['title']==title].index[0]
        query_emb = self.embeddings[idx].reshape(1, -1)

        similarities = cosine_similarity(query_emb, self.embeddings).flatten()
        top_indices = similarities.argsort()[::-1][1:top_n+1]

        return self.df.iloc[top_indices][['title', 'speaker']]

if __name__ == "__main__":
    rec = TEDRecommender("data/ted_talks.csv")
    print(rec.recommend("The Power of Vulnerability"))
