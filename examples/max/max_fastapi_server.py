"""
FastAPI In-Process Semantic Search Server with Modular MAX Pipelines & MojoVec.

Demonstrates serving a real embedding model (sentence-transformers/all-MiniLM-L6-v2)
directly compiled by Modular MAX (`max.pipelines`), indexing and querying vectors
in MojoVec in the same process with zero network hop latency.
"""

from contextlib import asynccontextmanager
from typing import List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from max.pipelines import PIPELINE_REGISTRY, PipelineArgs, PipelineConfig
from max.pipelines.modeling.types import (
    EmbeddingsGenerationInputs,
    PipelineTask,
    RequestID,
    TextGenerationRequest,
)
import mojovec


# Data models
class DocumentItem(BaseModel):
    id: int
    text: str
    category: Optional[str] = "general"


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class SearchResult(BaseModel):
    id: int
    distance: float
    document: str
    metadata: dict


# Global application state
state = {}


async def max_encode_single(text: str) -> np.ndarray:
    """Encode a single text prompt using MAX EmbeddingsPipeline."""
    tokenizer = state["tokenizer"]
    pipeline = state["pipeline"]
    model_name = state["model_name"]

    context = await tokenizer.new_context(
        TextGenerationRequest(
            request_id=RequestID(),
            prompt=text,
            model_name=model_name,
        )
    )
    pipeline_input = EmbeddingsGenerationInputs([{context.request_id: context}])
    output = pipeline.execute(pipeline_input)
    vec = output[context.request_id].embeddings
    norm = np.linalg.norm(vec)
    return vec / max(norm, 1e-12)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading & compiling MAX Pipeline for '{model_path}'...")
    config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(
            model_path=model_path,
            quantization_encoding="float32",
            task="embeddings_generation",
        )
    )
    tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(
        config, task=PipelineTask.EMBEDDINGS_GENERATION
    )

    state["tokenizer"] = tokenizer
    state["pipeline"] = pipeline
    state["model_name"] = model_path

    print("Initializing MojoVec Collection (384-dim, cosine metric)...")
    state["db"] = mojovec.Collection(dimension=384, metric="cosine", name="max_server_kb")
    yield
    state.clear()


app = FastAPI(
    title="MojoVec + MAX Semantic Search API",
    description="High-performance In-Process RAG combining Modular MAX Pipelines and MojoVec vector store.",
    lifespan=lifespan,
)


@app.post("/documents", status_code=201)
async def add_documents(items: List[DocumentItem]):
    """Embed and index documents into MojoVec using MAX."""
    if not items:
        raise HTTPException(status_code=400, detail="Document list cannot be empty")

    db: mojovec.Collection = state["db"]

    ids = [item.id for item in items]
    texts = [item.text for item in items]
    metadatas = [{"category": item.category} for item in items]

    # Compute real embeddings through MAX
    embeddings = []
    for text in texts:
        vec = await max_encode_single(text)
        embeddings.append(vec)

    embeddings_matrix = np.array(embeddings, dtype=np.float32)

    # Store in MojoVec
    db.upsert(
        ids=ids,
        embeddings=embeddings_matrix,
        metadatas=metadatas,
        documents=texts,
    )

    return {"status": "indexed", "count": len(items)}


@app.post("/search", response_model=List[SearchResult])
async def search_documents(req: QueryRequest):
    """Embed query via MAX and search nearest neighbors in MojoVec."""
    db: mojovec.Collection = state["db"]

    if db.count() == 0:
        return []

    # Encode query with MAX
    q_emb = await max_encode_single(req.query)
    q_matrix = q_emb.reshape(1, -1)

    # Query MojoVec
    results = db.query(query_embeddings=q_matrix, n_results=req.top_k)

    matches = []
    for match_id, dist, meta, doc in zip(
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0],
        results["documents"][0],
    ):
        if match_id >= 0:
            matches.append(
                SearchResult(
                    id=match_id,
                    distance=float(dist),
                    document=doc,
                    metadata=meta,
                )
            )

    return matches


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
