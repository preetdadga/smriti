from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Industry-standard defaults for general-purpose RAG:
# 1000 chars/chunk balances context richness vs. retrieval precision,
# 150 overlap (~15%) preserves continuity across chunk boundaries.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def chunk_documents(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)