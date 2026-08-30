"""
Knowledge Base using MAX Serve & MojoVec Vector Search.

This example follows the official Modular MAX serving architecture:
1. Start MAX Serve endpoint with an embedding model:
   $ max serve --model sentence-transformers/all-mpnet-base-v2 (or all-MiniLM-L6-v2)
2. This script sends documents to MAX Serve (/v1/embeddings), retrieves real dense
   vectors, indexes them into MojoVec, and runs semantic search queries.
"""

import logging
from typing import List, Tuple
import numpy as np
import requests
import mojovec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MaxServeKnowledgeBase:
    def __init__(
        self,
        endpoint: str = "http://localhost:8000/v1/embeddings",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
    ):
        self.endpoint = endpoint
        self.model_name = model_name
        self.dimension = dimension
        self.collection = mojovec.Collection(
            dimension=dimension,
            metric="cosine",
            name="max_serve_kb",
        )
        self.next_id = 1

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Fetch embeddings from the running MAX Serve endpoint."""
        response = requests.post(
            self.endpoint,
            headers={"Content-Type": "application/json"},
            json={
                "input": texts,
                "model": self.model_name,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return np.array([item["embedding"] for item in data["data"]], dtype=np.float32)

    def add_documents(self, titles: List[str], contents: List[str]):
        """Embed and index multiple documents into MojoVec."""
        embeddings = self.get_embeddings(contents)
        ids = list(range(self.next_id, self.next_id + len(contents)))
        self.next_id += len(contents)

        metadatas = [{"title": t} for t in titles]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=contents,
        )
        logger.info(f"Indexed {len(contents)} documents into MojoVec.")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Query the knowledge base using MAX Serve embeddings + MojoVec search."""
        query_vector = self.get_embeddings([query])
        results = self.collection.query(query_embeddings=query_vector, n_results=top_k)

        matches = []
        for dist, meta, doc in zip(
            results["distances"][0],
            results["metadatas"][0],
            results["documents"][0],
        ):
            matches.append((meta.get("title", "Untitled"), doc, dist))
        return matches


if __name__ == "__main__":
    print(__doc__)
    print("Example initialization:")
    kb = MaxServeKnowledgeBase()
    print("MojoVec Collection ready for MAX Serve integration.")
