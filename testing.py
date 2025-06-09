from rag_engine import get_rag_response

test_query = {"query": "What was Manoj doing on 80ft road?"}
result = get_rag_response(test_query)

print(result)