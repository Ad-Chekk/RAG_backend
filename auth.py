# auth.py

users = {
    "Sales_dept": {"password": "password123", "role": "finance"},
    "finance_dept": {"password": "letmein", "role": "legal"},
    "Admin": {"password": "adminpass", "role": "admin"},
}

def authenticate(username, password):
    user = users.get(username)
    if user and user["password"] == password:
        return {"username": username, "role": user["role"]}
    return None
