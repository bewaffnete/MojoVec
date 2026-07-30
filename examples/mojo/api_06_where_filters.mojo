"""
Typed metadata filtering with automatic bitmap indexes.

This example demonstrates:

- String, Int, Float64, and Bool predicates;
- `eq`, `ne`, `gt`, `gte`, `lt`, and `lte` operators;
- `in_` membership with an explicitly typed value list;
- nested `and_`, `or_`, and `not_` expressions;
- filtered nearest-neighbor queries through the normal Collection API;
- missing-field behavior.

Run from the repository root:

    mojo run -I . examples/mojo/api_06_where_filters.mojo
"""

from std.collections import List

from mojovec import Client, Metadata, QueryResults, Where


comptime DIMENSION = 4


def append_vector(mut values: List[Float32], value: Float32):
    for _ in range(DIMENSION):
        values.append(value)


def make_metadata(
    category: String,
    year: Int,
    score: Float64,
    published: Bool,
) -> Metadata:
    var result = Metadata()
    result.set("category", category)
    result.set("year", year)
    result.set("score", score)
    result.set("published", published)
    return result^


def print_ids(title: String, results: QueryResults):
    print("\n" + title)
    for record_id in results.ids[0]:
        if record_id >= 0:
            print("  id =", record_id)


def main() raises:
    var client = Client()
    var collection = client.create_collection(
        "typed_where",
        dimension=DIMENSION,
        M=8,
        ef_construction=64,
        ef_search=64,
        quantized=False,
    )

    var ids = [10, 20, 30, 40]
    var embeddings = List[Float32]()
    append_vector(embeddings, 0.0)
    append_vector(embeddings, 10.0)
    append_vector(embeddings, 20.0)
    append_vector(embeddings, 30.0)

    var metadatas = List[Metadata]()
    metadatas.append(make_metadata("docs", 2022, 0.70, True))
    metadatas.append(make_metadata("docs", 2025, 0.90, False))
    metadatas.append(make_metadata("code", 2024, 0.95, True))
    metadatas.append(make_metadata("code", 2026, 0.80, False))
    collection.add(ids, embeddings, metadatas)

    var query = List[Float32]()
    append_vector(query, 15.0)

    var docs = collection.query(
        query,
        where=Where.eq("category", "docs"),
        n_results=4,
    )
    print_ids("category == docs", docs)

    # Membership lists are typed, avoiding String/Int/Float ambiguity.
    var selected_years = List[Int]()
    selected_years.append(2022)
    selected_years.append(2026)
    var selected = collection.query(
        query,
        where=Where.in_("year", selected_years),
        n_results=4,
    )
    print_ids("year in [2022, 2026]", selected)

    # Nested logic: published AND (recent OR high score).
    var alternatives = List[Where]()
    alternatives.append(Where.gte("year", 2025))
    alternatives.append(Where.gt("score", Float64(0.92)))

    var requirements = List[Where]()
    requirements.append(Where.eq("published", True))
    requirements.append(Where.or_(alternatives))

    var compound = collection.query(
        query,
        where=Where.and_(requirements),
        n_results=4,
    )
    print_ids("published AND (recent OR high score)", compound)

    # `ne` requires the field to exist. `not_(eq(...))` negates the complete
    # expression and would also include records missing "category".
    var not_docs = collection.query(
        query,
        where=Where.ne("category", "docs"),
        n_results=4,
    )
    print_ids("category != docs", not_docs)
