from .collection import Collection
from .collection_ivfpq import CollectionIVFPQ

struct Client:
    def __init__(out self):
        pass

    def create_collection(self, name: String, dimension: Int, M: Int = 32, ef_construction: Int = 40, ef_search: Int = 16) -> Collection:
        """
        Creates a new vector collection with the specified name and dimension.
        """
        return Collection(dimension, M, ef_construction, ef_search)

    def create_ivfpq_collection(self, name: String, dimension: Int, nlist: Int = 100, M: Int = 16) -> CollectionIVFPQ:
        """
        Creates a new IVF-PQ vector collection for extreme compression.
        """
        return CollectionIVFPQ(dimension, nlist, M)
