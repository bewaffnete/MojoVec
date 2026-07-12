"""
Defines core traits and interfaces for the vector index and quantization.
"""

trait Index:
    """
    Defines the abstract interface for a vector search index.
    """
    def add(mut self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin]):
        """
        Adds multiple vectors to the index.
        
        Args:
            n: The number of vectors to add.
            x: A pointer to a flattened array of size `n * d` containing the vectors.
        """
        ...
        
    def search(self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin], k: Int, distances: UnsafePointer[Float32, MutUntrackedOrigin], labels: UnsafePointer[Int, MutUntrackedOrigin]):
        """
        Searches for the `k` nearest neighbors for `n` query vectors.
        
        Args:
            n: The number of query vectors.
            x: A pointer to a flattened array of size `n * d` containing the queries.
            k: The number of nearest neighbors to retrieve per query.
            distances: An output array of size `n * k` to store the resulting distances.
            labels: An output array of size `n * k` to store the resulting vector IDs.
        """
        ...

trait QuantizerTrait(Movable, ImplicitlyDeletable):
    """
    Defines the interface for a quantizer capable of encoding and decoding vectors.
    """
    def add(mut self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin]):
        """
        Adds multiple vectors to the quantizer.
        
        Args:
            n: The number of vectors.
            x: A pointer to a flattened array containing the vectors.
        """
        ...
        
    def search(self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin], k: Int, distances: UnsafePointer[Float32, MutUntrackedOrigin], labels: UnsafePointer[Int, MutUntrackedOrigin]):
        """
        Searches for the `k` nearest neighbors within the quantized vectors.
        
        Args:
            n: The number of query vectors.
            x: A pointer to a flattened array containing the queries.
            k: The number of nearest neighbors to retrieve.
            distances: An output array to store the resulting distances.
            labels: An output array to store the resulting vector IDs.
        """
        ...
        
    def get_vector(self, id: Int) -> UnsafePointer[Float32, MutUntrackedOrigin]:
        """
        Retrieves a vector by its ID.
        
        Args:
            id: The unique identifier of the vector.
            
        Returns:
            A pointer to the corresponding vector.
        """
        ...
