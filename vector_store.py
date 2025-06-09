# vector_store.py

import pickle
import numpy as np
import faiss

# Load FAISS index
index = faiss.read_index("data/df_faiss_index.index")  # Or "index.faiss" if you renamed

# Load metadata
with open("data/df_doc_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)  # Should be a list of dicts with 'access_roles'

# Load embeddings (not always needed, but useful for debugging)
embeddings = np.load("data/df_embeddings.npy")

def search(query_vector, user_role, top_k=5):
    D, I = index.search(np.array([query_vector]), top_k)
    results = []

    for idx in I[0]:
        if idx < len(metadata):
            doc = metadata[idx]
            if user_role in doc.get("access_roles", []):
                results.append(doc)

    return results
