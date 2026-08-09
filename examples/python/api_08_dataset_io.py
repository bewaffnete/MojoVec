"""Import CSV, Parquet, Arrow IPC, and Hugging Face datasets."""

from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as parquet

import mojovec


LOCAL_DIMENSION = 3
HF_DATASET = "christophsonntag/gte_embedded_movies"
HF_DIMENSION = 1024


def local_rows() -> list[dict[str, Any]]:
    """Return records shared by all three local format examples."""

    return [
        {
            "id": 10,
            "embedding": [1.0, 0.0, 0.0],
            "document": "A guide to vector search",
            "topic": "search",
            "year": 2026,
        },
        {
            "id": 20,
            "embedding": [0.8, 0.2, 0.0],
            "document": "Working with Arrow and Parquet dataset files",
            "topic": "data",
            "year": 2025,
        },
    ]


def write_local_datasets(directory: Path) -> dict[str, Path]:
    """Create tiny input files so the example runs without external assets."""

    rows = local_rows()
    csv_path = directory / "documents.csv"
    parquet_path = directory / "documents.parquet"
    arrow_path = directory / "documents.arrow"

    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "embedding": repr(row["embedding"])})

    table = pa.Table.from_pylist(rows)
    parquet.write_table(table, parquet_path)
    with pa.OSFile(str(arrow_path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    return {
        "CSV": csv_path,
        "Parquet": parquet_path,
        "Arrow IPC": arrow_path,
    }


def import_local_dataset(label: str, path: Path) -> None:
    """Use one Collection.add_from call for every supported local format."""

    collection = mojovec.Collection(
        dimension=LOCAL_DIMENSION,
        quantized=True,
        metric="cosine",
        name=f"{label.lower()}-documents",
    )
    imported = collection.add_from(
        path,
        id_column="id",
        embedding_column="embedding",
        document_column="document",
        metadata_columns=["topic", "year"],
        batch_size=1,
    )
    result = collection.query([[1.0, 0.0, 0.0]], n_results=2)

    print(f"\n{label} imported: {imported}")
    for rank, (record_id, distance, metadata, document) in enumerate(
        zip(
            result["ids"][0],
            result["distances"][0],
            result["metadatas"][0],
            result["documents"][0],
        ),
        start=1,
    ):
        print(
            f" {rank}. id={record_id} distance={distance:.4f} "
            f"topic={metadata['topic']!r}: {document}"
        )


def import_hugging_face_dataset() -> None:
    """Stream a compact public HF dataset directly into a MojoVec collection."""

    collection = mojovec.Collection(
        dimension=HF_DIMENSION,
        M=16,
        ef_construction=100,
        ef_search=64,
        quantized=True,
        metric="cosine",
        name="hf-movies",
    )
    imported = collection.add_huggingface(
        HF_DATASET,
        split="train",
        embedding_column="plot_embedding",
        document_column="fullplot",
        metadata_columns=["title", "runtime"],
        batch_size=256,
        streaming=True,
    )
    results = collection.query(
        query_texts=["pirate ship treasure adventure"],
        n_results=5,
    )

    print(f"\nHugging Face dataset: {HF_DATASET}")
    print(f"Imported movies: {imported}")
    print("BM25 results:")
    for rank, (score, metadata, document) in enumerate(
        zip(
            results["scores"][0],
            results["metadatas"][0],
            results["documents"][0],
        ),
        start=1,
    ):
        excerpt = document.replace("\n", " ")[:180]
        print(f" {rank}. {metadata['title']} | score={score:.4f}")
        print(f"    {excerpt}...")


def main() -> None:
    with TemporaryDirectory(prefix="mojovec-dataset-example-") as directory:
        paths = write_local_datasets(Path(directory))
        for label, path in paths.items():
            import_local_dataset(label, path)

    # This downloads about 10 MB on the first run and then uses the standard
    # Hugging Face cache. The dataset already contains 1,024-D embeddings.
    import_hugging_face_dataset()

    # Other vector-only formats use the same API and generate IDs from
    # id_start. NPZ may additionally contain ids, documents, and metadata:
    # collection.upsert_from("embeddings.npy", id_start=1_000)
    # collection.add_from("sift_base.fvecs", id_start=0)
    # collection.add_from("records.npz", metadata_columns=["category"])


if __name__ == "__main__":
    main()
