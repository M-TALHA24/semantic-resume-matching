# HNSW Index Hyperparameters
# M: Number of bidirectional links per node. 
# Higher = better accuracy/recall, but uses more RAM.
M = 16 

# ef_construction: Accuracy during the index building phase. 
# Higher = better index quality, but building takes longer.
EF_CONSTRUCTION = 200 

# ef_search: The "beam width" for search. 
# Higher = higher recall (accuracy), but slower query time.
# This is the primary parameter you tune during evaluation.
EF_SEARCH = 100 

# Model & Data Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'  # Sentence-BERT model
DIMENSION = 384                 # Vector dimension for this specific model

# File Paths
RESUME_VECTORS_PATH = 'embeddings/resumes_vectors.npy'
INDEX_SAVE_PATH = 'hnsw_engine/index.bin'

# API Configuration[cite: 1]
API_HOST = '0.0.0.0'
API_PORT = 5000
DEFAULT_K = 5  # Number of top results to return[cite: 1]
