from collections import defaultdict

def get_ingested_documents(vectorstore) -> list[dict]:
    """Group all chunks currently in Chroma by source file, with counts."""
    raw = vectorstore.get()
    counts = defaultdict(int)
    for metadata in raw["metadatas"]:
        source = metadata.get("source", "unknown")
        counts[source] += 1
    return [{"source": src, "chunks": n} for src, n in sorted(counts.items())]


def delete_document(vectorstore, source: str) -> int:
    """Delete all chunks belonging to a given source file. Returns count deleted."""
    result = vectorstore.get(where={"source": source})
    ids = result["ids"]
    if ids:
        vectorstore.delete(ids=ids)
    return len(ids)


def clear_all_documents(vectorstore):
    """Wipe the entire collection. Caller must reconnect (get_vectorstore())
    afterward, since the underlying collection no longer exists."""
    vectorstore.delete_collection()