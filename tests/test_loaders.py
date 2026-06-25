from backend.ingestion.loaders import load_document

docs = load_document("tests/test_files/Preet_Dadga_Resume_.pdf")
print(f"Loaded {len(docs)} page(s)")
print(docs[0].page_content[:200])
print(docs[0].metadata)