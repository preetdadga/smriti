from backend.retrieval.vectorstore import get_vectorstore
from backend.retrieval.hybrid import get_hybrid_retriever

store = get_vectorstore()
hybrid = get_hybrid_retriever(store, k=3)

query = "What is the zero-cost constraint?"
results = hybrid.invoke(query)

print(f"Retrieved {len(results)} result(s)\n")
for i, r in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(r.page_content[:200])
    print(r.metadata)
    print()