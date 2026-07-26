from .collection import Collection
from .collection_ivfpq import CollectionIVFPQ

struct Client:
    """
    Client for managing vector collections.
    """

    def __init__(out self):
        """
        Initializes a new vector client.
        """
        pass

    def create_collection(
        self,
        name: String,
        dimension: Int,
        M: Int = 32,
        ef_construction: Int = 40,
        ef_search: Int = 16,
        quantized: Bool = True,
    ) -> Collection:
        """
        Creates a Flat or SQ8 HNSW collection.
        """
        return Collection(
            dimension,
            M,
            ef_construction,
            ef_search,
            quantized,
            name,
        )

    def create_ivfpq_collection(self, name: String, dimension: Int, nlist: Int = 100, M: Int = 16) -> CollectionIVFPQ:
        """
        Creates a new IVF-PQ vector collection for extreme compression.
        """
        return CollectionIVFPQ(dimension, nlist, M)
