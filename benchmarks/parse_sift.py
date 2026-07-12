import numpy as np
import os

def fvecs_to_bin(input_file, output_file, max_vectors=None):
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
        
    print(f"Converting {input_file} -> {output_file}...")
    a = np.fromfile(input_file, dtype='int32')
    d = a[0]
    a = a.reshape(-1, d + 1)
    if max_vectors is not None:
        a = a[:max_vectors]
    vectors = a[:, 1:].view('float32')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    vectors.tofile(output_file)
    print(f"Saved {vectors.shape[0]} vectors of dimension {d} to {output_file}")

if __name__ == "__main__":
    # Convert base vectors (db.bin, all 1M vectors for the benchmark to match ground truth)
    fvecs_to_bin("sift1m/sift_base.fvecs", "benchmarks/suite/db.bin", max_vectors=None)
    
    # Convert queries (queries.bin, all 10k vectors)
    fvecs_to_bin("sift1m/sift_query.fvecs", "benchmarks/suite/queries.bin")
    
    print("All conversions complete!")
