from backend.retrieval.vectorstore import get_vectorstore
from backend.retrieval.hybrid import get_hybrid_retriever

store = get_vectorstore()
hybrid = get_hybrid_retriever(store, k=25)

results = hybrid.invoke("What textbooks are recommended for this course?")
print(f"Retrieved {len(results)} candidate(s) before reranking\n")

for i, r in enumerate(results):
    snippet = r.page_content[:80].replace("\n", " ")
    print(f"{i+1}. [{r.metadata.get('source')} p.{r.metadata.get('page')}] {snippet}")