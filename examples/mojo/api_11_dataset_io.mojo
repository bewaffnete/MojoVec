"""Read a vector matrix with pure Mojo and add it through the managed API."""

from mojovec import Collection, read_csv
from std.collections import List


def main() raises:
    # Native CSV/TSV readers intentionally accept numeric vector matrices.
    # IDs are generated sequentially; Python readers cover named documents,
    # metadata, Hugging Face, Parquet, Arrow, JSON/JSONL, and NPZ payloads.
    var path = "/tmp/mojovec_dataset_example.csv"
    var writer = open(path, "w")
    writer.write("x,y,z\n1.0,0.0,0.0\n0.0,1.0,0.0\n0.8,0.2,0.0\n")
    writer.close()

    var dataset = read_csv(
        path,
        dimension=3,
        has_header=True,
        id_start=100,
    )
    print("rows:", dataset.count(), "dimension:", dataset.dimension)

    var collection = Collection(
        dimension=dataset.dimension,
        quantized=True,
        metric="cosine",
    )
    collection.add(dataset.ids, dataset.embeddings)

    var query = List[Float32](capacity=3)
    query.append(1.0)
    query.append(0.0)
    query.append(0.0)
    var results = collection.query(query, n_results=2)
    print("nearest IDs:", results.ids[0])

    # Binary datasets use the same managed result shape:
    # var sift = read_fvecs("sift_base.fvecs", id_start=0)
    # collection.add(sift.ids, sift.embeddings)
    # var truth = read_ivecs("sift_groundtruth.ivecs")
    # var vectors = read_npy_float32("embeddings.npy", id_start=10_000)
