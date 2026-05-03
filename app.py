import os
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from search_engine import SearchEngine

app = Flask(__name__)
CORS(app)  # This allows Shazaib's frontend to talk to your backend

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESUME_ROOT = os.path.join(BASE_DIR, "data", "raw", "resumes", "data", "data")


def build_resume_file_index():
    resume_files = {}
    for root, _, files in os.walk(RESUME_ROOT):
        for file_name in files:
            if file_name.lower().endswith(".pdf"):
                resume_id = os.path.splitext(file_name)[0]
                resume_files[int(resume_id)] = os.path.join(root, file_name)
    return resume_files


RESUME_FILES = build_resume_file_index()

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

@app.route('/', methods=['GET'])
def home():
    """Serves the minimal frontend for testing search."""
    return render_template('index.html')

@app.route('/resume/<int:resume_id>', methods=['GET'])
def view_resume(resume_id):
    """Serves the original PDF resume from the codebase."""
    resume_path = RESUME_FILES.get(resume_id)
    if not resume_path or not os.path.exists(resume_path):
        return jsonify({"error": "Resume not found"}), 404

    return send_file(resume_path, mimetype='application/pdf', as_attachment=False, download_name=f'{resume_id}.pdf')

@app.errorhandler(404)
def handle_not_found(error):
    """Serve the frontend at the root path even if the live process is stale."""
    if request.path == '/' or request.path == '/index.html':
        return render_template('index.html'), 200
    return jsonify({"error": "Not found"}), 404

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
