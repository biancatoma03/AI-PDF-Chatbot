import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.index = None
        self.chunks = []
    def create_index(self, chunks):
        if not chunks:
            raise ValueError("No chunks provided")
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        print(f"FAISS index created with {self.index.ntotal} vectors")
    def search(self, query, top_k=3):
        if self.index is None:
            raise ValueError("Index not created yet")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(
            query_embedding,
            k
        )
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[int(idx)]
            results.append({
                "page": chunk["page"],
                "text": chunk["text"],
                "score": float(score)
            })

        return results