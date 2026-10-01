
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Load the fine-tuned model
model = SentenceTransformer("cs-paper-model")
model.max_seq_length = 256

# Load FAISS index
index = faiss.read_index("index.faiss")

# Load paper information
with open("corpus.json", "r", encoding="utf-8") as f:
    papers = json.load(f)


def search(query, top_k=5):
    """Search for papers semantically similar to the query."""

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        np.asarray(query_embedding, dtype="float32"),
        top_k
    )

    results = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):
        results.append({
            "rank": rank,
            "score": float(score),
            "title": papers[idx]["title"],
            "abstract": papers[idx]["abstract"]
        })

    return results
