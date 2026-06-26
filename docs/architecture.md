# Smriti — Architecture

## Overview
RAG chatbot: ingests PDF/DOCX/TXT, retrieves context via hybrid 
(vector + BM25) search with reranking, generates answers via a 
local LLM (Ollama), with source citations.

## Data Flow
Ingestion: Document -> Loader -> Chunker -> Embedder -> ChromaDB
Query: Question -> [Vector Search + BM25] -> Reranker -> Top-k chunks 
-> LLM -> Answer + Citations

## Directory Structure
smriti/
├── backend/
│   ├── ingestion/   # loaders, chunking
│   ├── retrieval/   # vector store, BM25, reranker
│   ├── generation/  # LLM calls, prompts, citations
│   └── api/         # FastAPI routes
├── frontend/        # Streamlit UI
├── docs/
└── tests/

## Roadmap
0. Foundation
1. Document ingestion (PDF/DOCX/TXT, chunking)
2. Semantic search (Ollama embeddings + ChromaDB)
3. RAG generation (Ollama LLM, prompts, citations)
4. Production retrieval (BM25 + hybrid + reranking)
5. Conversational memory
6. Streamlit UX
7. Dockerized FastAPI deployment

## Tech Stack
- Loaders: LangChain document_loaders (PyPDFLoader, Docx2txtLoader)
- Chunking: LangChain RecursiveCharacterTextSplitter
- Embeddings: LangChain OllamaEmbeddings (nomic-embed-text)
- Vector store: LangChain Chroma wrapper
- Hybrid retrieval: LangChain EnsembleRetriever (BM25Retriever + vector retriever)
- Reranking: LangChain ContextualCompressionRetriever + 
  HuggingFaceCrossEncoder (or Cohere reranker)
- Generation: LangChain ChatOllama + custom prompt templates
- Memory: LangChain RunnableWithMessageHistory
- Orchestration: LCEL chains throughout

## Models
- Embedding: nomic-embed-text (via Ollama)
- Generation: qwen3:8b (via Ollama)

Chosen for local inference on consumer hardware (RTX 3050 6GB VRAM).
Qwen3 8B runs in "thinking" mode by default; disabled via `/no_think`
prefix in the system prompt to reduce generation latency for
extractive QA, where reasoning overhead isn't needed.

## Known Limitations
- Conversational memory is in-process (Python dict), not persisted — 
  history is lost on restart. Acceptable for single-session demo use; 
  would need Redis/SQLite for multi-user production deployment.
- Uses RunnableWithMessageHistory (LangChain Core), which is deprecated 
  in favor of LangGraph persistence. Chosen deliberately to showcase 
  LCEL composition; LangGraph would be the production-recommended path.
- PyPDFLoader extracts text in raw stream order, which can scramble 
  reading order for multi-column academic PDFs (especially LaTeX-
  generated two-column layouts with embedded figures). Single-column 
  PDFs extract reliably. A production system would need a layout-aware 
  extractor (e.g. PyMuPDF with column detection, or the `unstructured` 
  library) to handle two-column papers correctly.
  
## Future Enhancements
- RAGAS evaluation harness
- Streaming responses
- Multi-user support