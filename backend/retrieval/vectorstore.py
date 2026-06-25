from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

EMBEDDING_MODEL = "nomic-embed-text"
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "smriti_docs"

def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )

def add_documents(chunks: list[Document]):
    store = get_vectorstore()
    store.add_documents(chunks)
    return store