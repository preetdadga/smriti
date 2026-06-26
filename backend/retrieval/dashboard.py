from collections import defaultdict

def get_ingested_documents(vectorstore) -> list[dict]:
    """Group all chunks currently in Chroma by source file, with counts."""
    raw = vectorstore.get()
    counts = defaultdict(int)
    for metadata in raw["metadatas"]:
        source = metadata.get("source", "unknown")
        counts[source] += 1
    return [{"source": src, "chunks": n} for src, n in sorted(counts.items())]