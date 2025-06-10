
from flask_cors import CORS
#from flask_session 
from flask import Flask, request, jsonify, session
import numpy as np
from datetime import timedelta

import os
# from data_processing import df
from auth import authenticate
from vector_store import search
from rag_engine import get_rag_response

app = Flask(__name__)

app.permanent_session_lifetime = timedelta(days=1)

CORS(app, supports_credentials=True, origins="http://localhost:3000")

app.secret_key = "super-secret-key"
embeddings = np.load("data/df_embeddings.npy")  # This ensures we match the dimension

# 
def embed_query(text):
    # Random vector with same shape as the existing embeddings
    return np.random.rand(embeddings.shape[1]).astype("float32")

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    user = authenticate(data["username"], data["password"])
    if user:
        session.permanent = True  #  Force session to persist
        session["user"] = user    #  Store user info
        return jsonify({"message": "Login successful", "user": user})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user", None)
    return jsonify({"message": "Logged out"})

@app.route("/rag-query", methods=["POST"])
def rag_query():
    print("SESSION CONTENT:", session) 
    # if "user" not in session:
    #     return jsonify({"error": "Unauthorized"}), 403

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

    # Convert first 10 rows of CSV to JSON 
    # sample_data = df.head(10).to_dict(orient="records")
    # return jsonify({"rows": sample_data})
    return jsonify("hello")
 


###this should be changed when using https this is only compatible for localhost 
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",  # default works for most use cases
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_HTTPONLY=True,
)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
    port = int(os.environ.get("PORT", 5000))  # use PORT env if available (Render sets it), else default to 5000
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)


 
    
