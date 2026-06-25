from backend.ingestion.loaders import load_document
from backend.ingestion.chunker import chunk_documents

docs = load_document("tests/test_files/sample.pdf")
chunks = chunk_documents(docs)

print(f"Original: {len(docs)} document(s)")
print(f"Chunked: {len(chunks)} chunk(s)")
print("\n--- First chunk ---")
print(chunks[0].page_content)
print(chunks[0].metadata)

if len(chunks) > 1:
    print("\n--- Second chunk (check overlap) ---")
    print(chunks[1].page_content[:200])