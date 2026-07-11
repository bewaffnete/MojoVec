import mojovec

print("Loaded mojovec!")
collection = mojovec.Collection(128, 32, 200, 40)
print("Created collection!")

ids = [1, 2, 3]
embeddings = [0.1] * (128 * 3)

collection.upsert_batch(ids, embeddings)
print("Upserted batch!")

res = collection.query_batch(embeddings[:128], 3)
print("Query result:")
print("IDs:", res["ids"])
print("Distances:", res["distances"])
