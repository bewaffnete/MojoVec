# Modular MAX + MojoVec Integration Examples

This directory demonstrates how to build high-performance Retrieval-Augmented Generation (RAG) and Semantic Search pipelines by combining **[Modular MAX](https://www.modular.com/max)** (AI inference & serving engine) with **[MojoVec](https://github.com/bewaffnete/MojoVec)** (SIMD-accelerated in-process vector database written in Mojo).

---

## 🌟 Why MAX + MojoVec?

- **Zero Network Overhead (In-Process RAG):** Both model execution (via Modular MAX) and vector indexing (via MojoVec) run in the same memory space. Embeddings produced by MAX are indexed directly without roundtrips to remote vector databases.
- **Hardware Acceleration:** MAX compiles and accelerates transformer models (CPU, Apple Silicon, NVIDIA/AMD GPUs), while MojoVec accelerates vector similarity calculation via hardware SIMD intrinsics.
- **Pure Ecosystem Synergies:** Both technologies are engineered for high throughput and predictable low latency.

---

## 📂 Examples Overview

| File | Description |
|---|---|
| [`max_embeddings_rag.py`](max_embeddings_rag.py) | Standalone In-Process RAG pipeline using native `max.pipelines` to encode text and `mojovec.Collection` to perform semantic search. |
| [`max_fastapi_server.py`](max_fastapi_server.py) | Production-ready FastAPI web service embedding and querying documents in-memory. |
| [`max_serve_kb.py`](max_serve_kb.py) | Client example connecting to `max serve` OpenAI-compatible `/v1/embeddings` endpoint with MojoVec indexing. |

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Ensure you have installed the project dependencies using [Pixi](https://pixi.sh):

```bash
# Install environment dependencies (Mojo, MAX, Python, MojoVec)
pixi install
```

If running on macOS, set the OpenMP duplicate library flag before running scripts that import PyTorch and MAX:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

---

### 2. Running the Examples

#### Example 1: Standalone In-Process RAG (`max_embeddings_rag.py`)

This script loads the `sentence-transformers/all-MiniLM-L6-v2` architecture through native `max.pipelines`, computes real float32 embeddings, indexes documents into a `mojovec.Collection`, and performs semantic queries:

```bash
pixi run python examples/max/max_embeddings_rag.py
```

**Expected Output:**
```text
Loading & compiling MAX Pipeline for 'sentence-transformers/all-MiniLM-L6-v2'...
Initializing MojoVec Collection (384-dim, cosine metric)...
Generating MAX embeddings for 5 documents...
Indexing documents into MojoVec...
Successfully indexed 5 documents into MojoVec!

Query: 'What is the high speed vector database built with Mojo?'
  -> [ID 1] Score (Distance): 0.1070
     Document: MojoVec is a high-performance SIMD-accelerated vector database written in Mojo.
     Metadata: {'category': 'database', 'lang': 'mojo'}
```

---

#### Example 2: FastAPI In-Process RAG Service (`max_fastapi_server.py`)

Run an HTTP server that hosts MAX embeddings and MojoVec in a single process:

```bash
pixi run python examples/max/max_fastapi_server.py
```

The server starts on `http://0.0.0.0:8000`.

**Index Documents:**
```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '[
    {"id": 1, "text": "MojoVec provides SIMD vector search in Mojo.", "category": "db"},
    {"id": 2, "text": "Modular MAX accelerates deep learning inference.", "category": "engine"}
  ]'
```

**Query Documents:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "high speed vector search", "top_k": 2}'
```

---

#### Example 3: Microservice Serving with `max serve` (`max_serve_kb.py`)

If you prefer a microservice architecture, MAX provides a dedicated serving binary (`max serve`):

1. **Start the MAX Serve endpoint in a separate terminal:**
   ```bash
   pixi run max serve --model sentence-transformers/all-MiniLM-L6-v2 --quantization-encoding float32
   ```

2. **Run the client knowledge base script:**
   ```bash
   pixi run python examples/max/max_serve_kb.py
   ```

---

## 🛠️ Code Snippet: In-Process Embeddings with MAX & MojoVec

```python
import asyncio
from max.pipelines import PIPELINE_REGISTRY, PipelineArgs, PipelineConfig
from max.pipelines.modeling.types import (
    EmbeddingsGenerationInputs,
    PipelineTask,
    RequestID,
    TextGenerationRequest,
)
import mojovec

# 1. Compile and instantiate MAX Embeddings Pipeline
config = PipelineConfig.from_args(
    PipelineArgs.from_flat_kwargs(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        quantization_encoding="float32",
        task="embeddings_generation",
    )
)
tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(
    config, task=PipelineTask.EMBEDDINGS_GENERATION
)

# 2. Generate embedding with MAX
async def get_embedding(text: str):
    ctx = await tokenizer.new_context(
        TextGenerationRequest(
            request_id=RequestID(),
            prompt=text,
            model_name="all-MiniLM-L6-v2",
        )
    )
    res = pipeline.execute(EmbeddingsGenerationInputs([{ctx.request_id: ctx}]))
    return res[ctx.request_id].embeddings

doc_embedding = asyncio.run(get_embedding("MojoVec is fast."))

# 3. Store in MojoVec
db = mojovec.Collection(dimension=384, metric="cosine")
db.add(ids=[1], embeddings=doc_embedding.reshape(1, -1))

# 4. Query MojoVec
query_embedding = asyncio.run(get_embedding("speedy vector db"))
results = db.query(query_embeddings=query_embedding.reshape(1, -1), n_results=1)
print("Top match ID:", results["ids"][0][0])
```
