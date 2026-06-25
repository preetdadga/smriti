from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}

def load_document(file_path: str) -> list[Document]:
    ext = Path(file_path).suffix.lower()
    if ext == ".txt":
        docs = TextLoader(file_path, autodetect_encoding=True).load()
        for doc in docs:
            doc.page_content = doc.page_content.lstrip("\ufeff")
        return docs
    loader_cls = LOADER_MAP.get(ext)
    if not loader_cls:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader_cls(file_path).load()