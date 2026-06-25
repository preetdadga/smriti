from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

def get_bm25_retriever(vectorstore, k: int = 3) -> BM25Retriever:
    """Rebuild a BM25 index from whatever is already persisted in Chroma."""
    raw = vectorstore.get()  # {'ids': [...], 'documents': [...], 'metadatas': [...]}

    docs = [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(raw["documents"], raw["metadatas"])
    ]

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25

def get_hybrid_retriever(vectorstore, k: int = 3, weights: list[float] = None):
    """Combine BM25 (keyword) + vector (semantic) retrieval via ensemble."""
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    bm25_retriever = get_bm25_retriever(vectorstore, k=k)

    weights = weights or [0.5, 0.5]  # BM25, vector — equal weight to start

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=weights,
    )