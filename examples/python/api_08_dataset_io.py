"""Import local files and Hugging Face datasets with the managed Python API."""

from __future__ import annotations

import csv
from pathlib import Path
from tempfile import gettempdir

import mojovec


def main() -> None:
    # Named-column files can carry IDs, vectors, documents, and scalar
    # metadata together. Embeddings may be JSON arrays in one column or one
    # scalar column per dimension.
    csv_path = Path(gettempdir()) / "mojovec_dataset_example.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["id", "embedding", "document", "topic", "year"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "id": 10,
                    "embedding": "[1.0, 0.0, 0.0]",
                    "document": "A guide to vector search",
                    "topic": "search",
                    "year": 2026,
                },
                {
                    "id": 20,
                    "embedding": "[0.0, 1.0, 0.0]",
                    "document": "Working with dataset files",
                    "topic": "data",
                    "year": 2025,
                },
            ]
        )

    collection = mojovec.Collection(
        dimension=3,
        quantized=True,
        metric="cosine",
        name="imported-documents",
    )
    imported = collection.add_from(
        csv_path,
        id_column="id",
        embedding_column="embedding",
        document_column="document",
        metadata_columns=["topic", "year"],
        batch_size=1,
    )
    print("Imported rows:", imported)
    print(collection.query([[1.0, 0.0, 0.0]], n_results=2))

    # NumPy and vecs files have no named payload columns. IDs are generated
    # from id_start, and contiguous float32 batches use the numeric fast path:
    #
    # collection.upsert_from("embeddings.npy", id_start=1_000)
    # collection.add_from("sift_base.fvecs", id_start=0)
    #
    # NPZ can additionally provide arrays named embeddings, ids, documents,
    # plus any arrays selected by metadata_columns.

    # Hugging Face Datasets is optional and streams by default. Install it
    # with: pip install "mojovec[huggingface]"
    #
    # collection.add_huggingface(
    #     "owner/dataset",
    #     split="train",
    #     id_column="id",
    #     embedding_column="embedding",
    #     document_column="text",
    #     metadata_columns=["category"],
    #     batch_size=8192,
    # )


if __name__ == "__main__":
    main()
