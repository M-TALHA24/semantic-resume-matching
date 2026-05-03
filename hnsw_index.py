import hnswlib
import numpy as np
import os

class HNSWIndex:
    def __init__(self, dimension=384, max_elements=1000):
        # dimension: 384 for all-MiniLM-L6-v2
        self.dimension = dimension
        self.max_elements = max_elements
        
        # Initializing the index using L2 (Euclidean) or Cosine space
        # HNSWlib uses 'l2', 'ip' (inner product), or 'cosine'
        self.p = hnswlib.Index(space='cosine', dim=self.dimension)

    def build_index(self, data_vectors, M=16, ef_construction=200):
        """
        M: Links per node. Higher = better accuracy but more memory.
        ef_construction: Accuracy during build. Higher = better quality.
        """
        # Initialize the index with specific hyperparameters
        self.p.init_index(max_elements=self.max_elements, ef_construction=ef_construction, M=M)
        
        # Create IDs for the vectors (0 to N-1)
        ids = np.arange(len(data_vectors))
        
        # Add vectors to the index[cite: 1]
        self.p.add_items(data_vectors, ids)
        
        # Set ef_search for future queries (can be tuned later)[cite: 1]
        self.p.set_ef(50) 
        print(f"Index built with {self.p.element_count} elements.")

    def save_index(self, path="hnsw_engine/index.bin"):
        self.p.save_index(path)
        print(f"Index saved to {path}")

# Example usage for your local testing:
if __name__ == "__main__":
    # In reality, you will load 'resumes_vectors.npy' from Talha[cite: 1]
    # Here we create dummy data to ensure the script runs
    example_data = np.random.random((100, 384)).astype('float32')
    
    indexer = HNSWIndex(dimension=384, max_elements=100)
    indexer.build_index(example_data)
    
    if not os.path.exists("hnsw_engine"):
        os.makedirs("hnsw_engine")
    indexer.save_index()
