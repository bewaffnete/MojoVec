"""
Defines core traits and interfaces for the vector index and quantization.
"""

trait Index:
    """
    Defines the abstract interface for a vector search index.
    """
    def add(mut self, x: Span[Float32, _]):
        """
        Adds multiple vectors to the index.
        
        Args:
            x: A safe Span pointing to the flattened vectors to add.
        """
        ...
        
    def search(self, x: Span[Float32, _], k: Int, mut distances: Span[mut=True, Float32, _], mut labels: Span[mut=True, Int, _]):
        """
        Searches for the `k` nearest neighbors for the query vectors.
        
        Args:
            x: A safe Span pointing to the flattened query vectors.
            k: The number of nearest neighbors to retrieve per query.
            distances: An output Span to store the resulting distances.
            labels: An output Span to store the resulting vector IDs.
        """
        ...

trait QuantizerTrait(Movable, ImplicitlyDeletable):
    """
    Defines the interface for a quantizer capable of encoding and decoding vectors.
    """
    def add(mut self, x: Span[Float32, _]):
        """
        Adds multiple vectors to the quantizer.
        
        Args:
            x: A safe Span pointing to the flattened vectors to add.
        """
        ...
        
    def search(self, x: Span[Float32, _], k: Int, mut distances: Span[mut=True, Float32, _], mut labels: Span[mut=True, Int, _]):
        """
        Searches for the `k` nearest neighbors within the quantized vectors.
        
        Args:
            x: A safe Span pointing to the flattened query vectors.
            k: The number of nearest neighbors to retrieve.
            distances: An output Span to store the resulting distances.
            labels: An output Span to store the resulting vector IDs.
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
