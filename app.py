from flask import Flask, request, jsonify
from flask_cors import CORS
from search_engine import SearchEngine

app = Flask(__name__)
CORS(app)  # This allows Shazaib's frontend to talk to your backend

# Initialize your search engine
# It will load the model and the index you built earlier
try:
    engine = SearchEngine(index_path="hnsw_engine/index.bin")
    print("Search Engine loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load index. Error: {e}")
    engine = None

@app.route('/health', methods=['GET'])
def health():
    """Confirms the API is running."""
    return jsonify({"status": "healthy", "engine_loaded": engine is not None})

@app.route('/search', methods=['POST'])
def search():
    """Accepts job description, returns top-K matched resumes."""
    if engine is None:
        return jsonify({"error": "Search engine not initialized"}), 500

    # Get data from the frontend request
    data = request.json
    job_description = data.get('job_description', '')
    top_k = data.get('k', 5)

    if not job_description:
        return jsonify({"error": "No job description provided"}), 400

    # Perform the search using your search_engine.py logic
    labels, scores = engine.search(job_description, k=top_k)

    # Format the results for Shazaib's frontend
    results = []
    for label, score in zip(labels, scores):
        results.append({
            "resume_id": int(label),
            "similarity_score": float(round(score, 4))
        })

    return jsonify({
        "query": job_description,
        "results": results
    })

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
