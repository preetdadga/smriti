from backend.ingestion.loaders import load_document
from backend.ingestion.chunker import chunk_documents
from backend.retrieval.vectorstore import add_documents, get_vectorstore

docs = load_document("tests/test_files/sample.pdf")
chunks = chunk_documents(docs)
print(f"Chunked into {len(chunks)} pieces")

store = add_documents(chunks)
print("Added to Chroma")

results = store.similarity_search("what is this document about", k=3)
print(f"\nRetrieved {len(results)} result(s)")
for i, r in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(r.page_content[:200])
    print(r.metadata)