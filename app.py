# app.py

from flask import Flask, request, jsonify, session
import numpy as np

from data_processing import df
from auth import authenticate
from vector_store import search
from rag_engine import get_rag_response

app = Flask(__name__)
app.secret_key = "super-secret-key"
embeddings = np.load("data/df_embeddings.npy")  # This ensures we match the dimension

# ✅ Replace this with your real embedding model if needed
def embed_query(text):
    # Random vector with same shape as the existing embeddings
    return np.random.rand(embeddings.shape[1]).astype("float32")

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    user = authenticate(data["username"], data["password"])
    if user:
        session["user"] = user
        return jsonify({"message": "Login successful", "user": user})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user", None)
    return jsonify({"message": "Logged out"})

@app.route("/rag-query", methods=["POST"])
def rag_query():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 403

    # query = request.json.get("query", "")
    # query_vector = embed_query(query)
    # user_role = session["user"]["role"]

    # results = search(query_vector, user_role)

    query = request.get_json()
    answer = get_rag_response(query)
    return answer



@app.route("/view-roles", methods=["GET"])
def view_roles():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 403

    username = session["user"]["username"]

    # Only 'Admin' is allowed to view roles.csv
    if username != "Admin":
        return jsonify({"error": "Access denied"}), 403

    # Convert first 10 rows of CSV to JSON (don't send all 26k!)
    sample_data = df.head(10).to_dict(orient="records")
    return jsonify({"rows": sample_data})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
