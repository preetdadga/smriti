from backend.retrieval.vectorstore import get_vectorstore
from backend.retrieval.hybrid import get_hybrid_retriever
from backend.retrieval.reranker import get_reranked_retriever
from backend.generation.chain import answer_question

store = get_vectorstore()
hybrid = get_hybrid_retriever(store, k=5)
retriever = get_reranked_retriever(hybrid, top_n=3)

session_id = "test-session-1"

print("--- Turn 1 ---")
result1 = answer_question("What is the zero-cost constraint?", retriever, session_id)
print(result1["answer"])

print("\n--- Turn 2 (follow-up) ---")
result2 = answer_question("What did I just ask you about?", retriever, session_id)
print(result2["answer"])