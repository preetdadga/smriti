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

## Future Enhancements
- RAGAS evaluation harness
- Streaming responses
- Multi-user support