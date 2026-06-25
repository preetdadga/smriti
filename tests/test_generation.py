from backend.retrieval.vectorstore import get_vectorstore
from backend.retrieval.hybrid import get_hybrid_retriever
from backend.retrieval.reranker import get_reranked_retriever
from backend.generation.chain import answer_question

store = get_vectorstore()
hybrid = get_hybrid_retriever(store, k=5)
retriever = get_reranked_retriever(hybrid, top_n=3)

result = answer_question("What is the zero-cost constraint?", retriever)

print("--- Answer ---")
print(result["answer"])
print("\n--- Sources ---")
for s in result["sources"]:
    print(s)