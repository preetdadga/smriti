from backend.ingestion.loaders import load_document

pdf_docs = load_document("tests/test_files/Preet_Dadga_Resume_.pdf")
print(f"PDF: loaded {len(pdf_docs)} page(s)")
print(pdf_docs[0].page_content[:200])

docx_docs = load_document("tests/test_files/Sample.docx")  # match your actual filename
print(f"\nDOCX: loaded {len(docx_docs)} page(s)")
print(docx_docs[0].page_content[:200])
print(docx_docs[0].metadata)