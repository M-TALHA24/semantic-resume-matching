import numpy as np
import os
from hnsw_index import HNSWIndex
from search_engine import SearchEngine
from config import RESUME_VECTORS_PATH
from config import INDEX_SAVE_PATH
def main():
    
    # Create the directory for the index if it doesn't exist
    if not os.path.exists("hnsw_engine"):
        os.makedirs("hnsw_engine")

    print("--- Step 1: Loading Embeddings from Talha ---")
    if not os.path.exists(RESUME_VECTORS_PATH):
        print(f"Error: {RESUME_VECTORS_PATH} not found. Talha needs to generate this first!")
        return
    
    vectors = np.load(RESUME_VECTORS_PATH).astype('float32')
    num_vectors, dimension = vectors.shape
    print(f"Loaded {num_vectors} vectors with dimension {dimension}.")

    print("\n--- Step 2: Building the HNSW Index ---")
    # Initialize index with the correct dimensions (384 for BERT)
    indexer = HNSWIndex(dimension=dimension, max_elements=num_vectors)
    
    # M=16 and ef_construction=200 are standard starting points
    indexer.build_index(vectors, M=16, ef_construction=200)
    indexer.save_index(INDEX_SAVE_PATH)

    print("\n--- Step 3: Verifying Search Engine Logic ---")
    # Test if the search engine can load the newly created index
    try:
        engine = SearchEngine(index_path=INDEX_SAVE_PATH)
        test_query = "Software Engineer with Python experience"
        
        print(f"Testing Query: '{test_query}'")
        labels, scores = engine.search(test_query, k=3)
        
        print("Top Matches Found:")
        for label, score in zip(labels, scores):
            print(f" - Resume ID: {label} (Confidence: {score:.4f})")
            
    except Exception as e:
        print(f"Verification failed: {e}")

    print("\n--- System Ready ---")
    print("You can now run 'python app.py' to start the API for Shazaib.")

if __name__ == "__main__":
    main()
