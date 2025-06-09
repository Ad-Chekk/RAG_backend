# rag_engine.py

import os
import numpy as np
import pickle
import faiss

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer

# === Paths ===
DATA_DIR = "./data"
EMBEDDING_FILE = os.path.join(DATA_DIR, "df_embeddings.npy")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "df_faiss_index.index")
METADATA_FILE = os.path.join(DATA_DIR, "df_doc_metadata.pkl")

# === Load RAG Components ===
print("Loading embeddings, FAISS index, and metadata...")
embeddings = np.load(EMBEDDING_FILE)
index = faiss.read_index(FAISS_INDEX_FILE)

with open(METADATA_FILE, "rb") as f:
    doc_metadata = pickle.load(f)  # This is a DataFrame

# Rebuild lookup dictionaries from DataFrame
doc_mapping = dict(zip(range(len(doc_metadata)), doc_metadata["DOC_ID"].tolist()))
id_to_text = dict(zip(range(len(doc_metadata)), doc_metadata["combined_text"].tolist()))
id_to_title = dict(zip(range(len(doc_metadata)), doc_metadata["DOC_TITL"].tolist()))

# === Load Models ===
print("Loading embedding and generation models...")
embedding_model = SentenceTransformer("intfloat/e5-large-v2")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
generator = pipeline("text2text-generation", model=gen_model, tokenizer=tokenizer, max_length=512)

# === Main RAG Function ===
def get_rag_response(query_json: dict, top_k: int = 3) -> dict:
    """
    Accepts query JSON: { "query": "your question" }
    Returns JSON: { "answer": ..., "context": ..., "documents": [...] }
    """
    query = query_json.get("query", "").strip()
    if not query:
        return {"error": "Query cannot be empty."}

    # === Embed the query ===
    query_embedding = embedding_model.encode(["query: " + query])
    faiss.normalize_L2(query_embedding)

    # === Search FAISS index ===
    scores, indices = index.search(query_embedding, top_k)

    # === Prepare Context ===
    retrieved_docs = []
    context_parts = []
    for i, idx in enumerate(indices[0]):
        doc_id = doc_mapping[idx]
        title = id_to_title[idx]
        text = id_to_text[idx]
        score = float(scores[0][i])

        retrieved_docs.append({
            "id": doc_id,
            "title": title,
            "score": round(score, 4)
        })

        context_parts.append(f"Document {i+1} (Score: {score:.4f})\nTitle: {title}\nContent: {text}\n")

    context_text = "\n".join(context_parts)

    # === Build Prompt ===
    prompt = f"Use the following documents to answer the question:\n\n{context_text}\nQuestion: {query}"

    # === Generate Answer ===
    try:
        answer = generator(prompt)[0]["generated_text"]
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    return {
        "question": query,
        "answer": answer,
        "context": context_text,
        "documents": retrieved_docs
    }
