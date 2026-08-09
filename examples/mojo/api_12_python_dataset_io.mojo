"""Use Python ecosystem readers while keeping collection operations in Mojo."""

from mojovec import Collection, DatasetImportOptions, read_json


def main() raises:
    var path = "/tmp/mojovec_python_dataset_example.json"
    var writer = open(path, "w")
    writer.write(
        '[{"id":10,"embedding":[1,0,0],"text":"Vector search'
        ' with Mojo","topic":"search"},{"id":20,'
        '"embedding":[0,1,0],"text":"Reading external datasets",'
        '"topic":"data"}]'
    )
    writer.close()

    # DatasetImportOptions is shared by JSON/JSONL, Parquet, Arrow, NPZ, and
    # Hugging Face readers. Empty optional column names mean "not present".
    var options = DatasetImportOptions()
    options.id_column = "id"
    options.document_column = "text"
    options.metadata_columns.append("topic")
    options.batch_size = 1

    # Python decodes the file one batch at a time. add_to() converts that batch
    # into managed Mojo Lists and calls the normal insert-only Collection.add.
    var reader = read_json(path, 3, options)
    var collection = Collection(3, quantized=True, metric="cosine")
    print("imported:", reader.add_to(collection))
    print("document 10:", collection.get_document(10))
    print("topic 10:", collection.get_metadata(10).get_string("topic"))

    # The other ecosystem readers have the same one-shot API:
    # var parquet = read_parquet("records.parquet", 3, options)
    # var arrow = read_arrow("records.arrow", 3, options)
    # var archive = read_npz("records.npz", 3, options)
    # var hf = read_huggingface("owner/dataset", 3, options)
    # _ = parquet.upsert_to(collection)
