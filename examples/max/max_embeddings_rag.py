"""
End-to-End In-Process RAG: Modular MAX Pipelines + MojoVec Vector Database.

This example loads an embedding model (sentence-transformers/all-MiniLM-L6-v2)
directly through Modular MAX Pipelines (`max.pipelines`), generates real 384-dim
dense embeddings, indexes them into MojoVec, and runs semantic search.
"""

import asyncio
from typing import List
import numpy as np
from max.pipelines import PIPELINE_REGISTRY, PipelineArgs, PipelineConfig
from max.pipelines.modeling.types import (
    EmbeddingsGenerationInputs,
    PipelineTask,
    RequestID,
    TextGenerationRequest,
)
import mojovec


class MaxEmbeddingEngine:
    """Wrapper around Modular MAX EmbeddingsPipeline."""

    def __init__(self, model_path: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_path = model_path
        print(f"Loading & compiling MAX Pipeline for '{model_path}'...")
        config = PipelineConfig.from_args(
            PipelineArgs.from_flat_kwargs(
                model_path=model_path,
                quantization_encoding="float32",
                task="embeddings_generation",
            )
        )
        self.tokenizer, self.pipeline = PIPELINE_REGISTRY.retrieve(
            config, task=PipelineTask.EMBEDDINGS_GENERATION
        )

    async def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string to a 1D float32 numpy vector using MAX."""
        context = await self.tokenizer.new_context(
            TextGenerationRequest(
                request_id=RequestID(),
                prompt=text,
                model_name=self.model_path,
            )
        )
        pipeline_input = EmbeddingsGenerationInputs([{context.request_id: context}])
        output = self.pipeline.execute(pipeline_input)
        return output[context.request_id].embeddings

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts using MAX into a 2D float32 matrix."""
        vectors = []
        for t in texts:
            vec = asyncio.run(self.encode_single(t))
            vectors.append(vec)
        # Normalize for cosine similarity if needed
        matrix = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, a_min=1e-12, a_max=None)


def main():
    # 1. Initialize Modular MAX embedding model
    max_engine = MaxEmbeddingEngine("sentence-transformers/all-MiniLM-L6-v2")

    # 2. Initialize MojoVec Collection
    print("\nInitializing MojoVec Collection (384-dim, cosine metric)...")
    collection = mojovec.Collection(
        dimension=384,
        metric="cosine",
        name="max_mojovec_kb",
    )

    documents = [
        "MojoVec is a high-performance SIMD-accelerated vector database written in Mojo.",
        "Modular MAX is a unified AI engine and serving platform for high-throughput inference.",
        "Retrieval-Augmented Generation (RAG) combines semantic vector search with LLMs.",
        "Python interop in Mojo allows seamless data exchange without network overhead.",
        "Cosine similarity measures the angular distance between normalized vector embeddings.",
    ]

    metadatas = [
        {"category": "database", "lang": "mojo"},
        {"category": "ai_engine", "vendor": "modular"},
        {"category": "ai_pattern", "type": "rag"},
        {"category": "interop", "tech": "mojo_python"},
        {"category": "math", "type": "metric"},
    ]

    ids = list(range(1, len(documents) + 1))

    # 3. Compute real embeddings via MAX and index into MojoVec
    print(f"Generating MAX embeddings for {len(documents)} documents...")
    embeddings = max_engine.encode(documents)

    print("Indexing documents into MojoVec...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )
    print(f"Successfully indexed {collection.count()} documents into MojoVec!\n")

    # 4. Perform semantic search queries
    queries = [
        "What is the high speed vector database built with Mojo?",
        "How do I serve AI models with Modular?",
        "Explain semantic search with language models",
    ]

    for q in queries:
        print(f"Query: '{q}'")
        q_emb = max_engine.encode([q])
        results = collection.query(query_embeddings=q_emb, n_results=2)

        for match_id, dist, meta, doc in zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
            results["documents"][0],
        ):
            if match_id >= 0:
                print(f"  -> [ID {match_id}] Score (Distance): {dist:.4f}")
                print(f"     Document: {doc}")
                print(f"     Metadata: {meta}")
        print()


if __name__ == "__main__":
    main()
