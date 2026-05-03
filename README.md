# Resume Matching

Semantic matching system that finds relevant resumes for job descriptions using transformer-based embeddings and HNSW vector search.

## Prerequisites

- Python 3.9+
- pip

## Setup

1. **Clone or navigate to the project directory.**

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Build the HNSW index** (if not already built):
   ```bash
   python main_system.py
   ```
   This generates the vector embeddings and saves the searchable index.

5. **Start the Flask server:**
   ```bash
   python app.py
   ```
   The server runs on `http://localhost:5000`.

6. **Open the frontend:**
   Navigate to `http://localhost:5000` in your browser and paste a job description to see matching resumes.

## Project Structure

```
.
├── app.py                    # Flask backend with search and resume serving
├── search_engine.py          # Semantic search using HNSW and embeddings
├── hnsw_index.py            # HNSW index construction and loading
├── main_system.py           # Entry point for building the index
├── config.py                # Configuration and paths
├── templates/
│   └── index.html           # Frontend interface
├── static/
│   └── styles.css           # Styling
├── data/
│   ├── processed/           # Cleaned CSV data
│   │   ├── resumes_cleaned.csv
│   │   └── jobs_cleaned.csv
│   └── raw/resumes/         # Original PDF resumes by category
│       └── data/data/<CATEGORY>/<ID>.pdf
├── embeddings/              # Generated embeddings and index
│   ├── resumes_vectors.npy
│   ├── job_vectors.npy
│   └── resume_ids.json
└── src/
    ├── embeddings.py        # Embedding generation
    └── preprocessing.py     # Text preprocessing
```

## Usage

### Via Web UI

1. Start the server: `python app.py`
2. Go to `http://localhost:5000`
3. Paste a job description, set top-K results, and click Search
4. Click any resume ID to view the original PDF

### Via API

**Health check:**
```bash
curl http://localhost:5000/health
```

**Search for matching resumes:**
```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"job_description": "Python developer with Flask experience", "k": 5}'
```

**Get a resume PDF:**
```bash
curl http://localhost:5000/resume/16852973 -o resume.pdf
```

## Notes

- The model used is `all-MiniLM-L6-v2` from Sentence Transformers (384-dimensional embeddings).
- HNSW index parameters: M=16, ef_construction=200, ef_search=100.
- Resumes are served as inline PDFs (open in browser) by default.
