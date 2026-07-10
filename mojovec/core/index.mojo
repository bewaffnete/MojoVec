
trait Index:
    """
    Abstract interface for a Vector Index.
    """
    def add(mut self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin]):
        """
        Add n vectors of dimension d to the index.
        x is a pointer to the flattened array of size n * d.
        """
        ...
        
    def search(self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin], k: Int, distances: UnsafePointer[Float32, MutUntrackedOrigin], labels: UnsafePointer[Int, MutUntrackedOrigin]):
        """
        Search for the k nearest neighbors for n query vectors.
        x is a pointer to the flattened queries array of size n * d.
        distances is the output array of size n * k.
        labels is the output array of size n * k containing the IDs.
        """
        ...

trait QuantizerTrait(Movable, ImplicitlyDeletable):
    def add(mut self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin]): ...
    def search(self, n: Int, x: UnsafePointer[Float32, MutUntrackedOrigin], k: Int, distances: UnsafePointer[Float32, MutUntrackedOrigin], labels: UnsafePointer[Int, MutUntrackedOrigin]): ...
    def get_vector(self, id: Int) -> UnsafePointer[Float32, MutUntrackedOrigin]: ...
