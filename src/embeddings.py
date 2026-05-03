import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


# PATHS
RESUME_PATH = "data/processed/resumes_cleaned.csv"
JOBS_PATH = "data/processed/jobs_cleaned.csv"

OUTPUT_DIR = "embeddings"
RESUME_EMB_PATH = os.path.join(OUTPUT_DIR, "resumes_vectors.npy")
JOB_EMB_PATH = os.path.join(OUTPUT_DIR, "job_vectors.npy")
RESUME_IDS_PATH = os.path.join(OUTPUT_DIR, "resume_ids.json")


# LOAD DATA
def load_data():
    resumes = pd.read_csv(RESUME_PATH)
    jobs = pd.read_csv(JOBS_PATH)

    resumes["Resume_str"] = resumes["Resume_str"].fillna("")
    jobs["job_str"] = jobs["job_str"].fillna("")

    print("Data loaded:")
    print("Resumes:", resumes.shape)
    print("Jobs:", jobs.shape)

    return resumes, jobs


# LOAD MODEL
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Sentence-BERT model loaded")
    return model


# CREATE EMBEDDINGS
def create_embeddings(model, texts, name="data"):
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"{name} embeddings shape:", embeddings.shape)
    return embeddings


# SAVE EMBEDDINGS
def save_embeddings(resume_vecs, job_vecs):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(RESUME_EMB_PATH, resume_vecs)
    np.save(JOB_EMB_PATH, job_vecs)

    print("Embeddings saved in /embeddings folder")


# SAVE IDS
def save_ids(resumes):
    resume_ids = resumes["ID"].tolist()
    with open(RESUME_IDS_PATH, "w") as f:
        json.dump(resume_ids, f)
    print(f"Resume IDs saved — {len(resume_ids)} IDs")


# MAIN PIPELINE
def main():
    resumes, jobs = load_data()
    model = load_model()

    resume_texts = resumes["Resume_str"].tolist()
    job_texts = jobs["job_str"].tolist()

    resume_vectors = create_embeddings(model, resume_texts, "Resume")
    job_vectors = create_embeddings(model, job_texts, "Job")

    assert resume_vectors.shape[0] == len(resumes)
    assert job_vectors.shape[0] == len(jobs)
    assert resume_vectors.shape[1] == 384

    print("Validation passed")

    save_embeddings(resume_vectors, job_vectors)
    save_ids(resumes)


if __name__ == "__main__":
    main()
