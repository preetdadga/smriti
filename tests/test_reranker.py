from backend.retrieval.vectorstore import get_vectorstore
from backend.retrieval.hybrid import get_hybrid_retriever
from backend.retrieval.reranker import get_reranked_retriever

store = get_vectorstore()
hybrid = get_hybrid_retriever(store, k=5)  # cast a wider net before reranking
reranked = get_reranked_retriever(hybrid, top_n=3)

query = "What is the zero-cost constraint?"
results = reranked.invoke(query)

print(f"Reranked to {len(results)} result(s)\n")
for i, r in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(r.page_content[:200])
    print(r.metadata)
    print()