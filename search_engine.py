import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer

class SearchEngine:
    def __init__(self, index_path="hnsw_engine/index.bin", model_name='all-MiniLM-L6-v2'):
        # 1. Load the same BERT model Talha is using for embeddings
        self.model = SentenceTransformer(model_name)
        
        # 2. Load the HNSW index you saved in the previous step
        self.dimension = 384 # Standard for this model
        self.p = hnswlib.Index(space='cosine', dim=self.dimension)
        self.p.load_index(index_path)
        
        # 3. Set ef_search: Higher = more accurate but slower
        # This is a key parameter Safiullah must know for the viva
        self.p.set_ef(100) 

    def search(self, job_description, k=5):
        """
        Converts text to vector and finds top-K matches.
        """
        # Convert the job description text into a 384-dim vector[cite: 1]
        query_vector = self.model.encode([job_description])
        
        # Query the HNSW index
        # labels: the IDs of the resumes; distances: how similar they are[cite: 1]
        labels, distances = self.p.knn_query(query_vector, k=k)
        
        # Convert distances to similarity scores (0 to 1 range)
        # For 'cosine' space in hnswlib, distance is (1 - similarity)
        scores = 1 - distances[0]
        
        return labels[0], scores

# Example usage for testing:
if __name__ == "__main__":
    # Ensure hnsw_engine/index.bin exists before running this[cite: 1]
    try:
        engine = SearchEngine()
        job_desc = "Looking for a Software Engineer proficient in Python and Flask."
        
        results, scores = engine.search(job_desc, k=3)
        
        print("Search Results:")
        for idx, score in zip(results, scores):
            print(f"Resume ID: {idx} | Similarity Score: {score:.4f}")
            
    except Exception as e:
        print(f"Error: {e}. Make sure you've built and saved the index first!")
