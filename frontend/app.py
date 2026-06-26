import streamlit as st
import sys
from pathlib import Path

# Allow imports from backend/ when running via `streamlit run`
sys.path.append(str(Path(__file__).parent.parent))

from backend.ingestion.loaders import load_document
from backend.ingestion.chunker import chunk_documents
from backend.retrieval.vectorstore import get_vectorstore, add_documents
from backend.retrieval.hybrid import get_hybrid_retriever
from backend.retrieval.reranker import get_reranked_retriever
from backend.retrieval.dashboard import get_ingested_documents, delete_document, clear_all_documents
from backend.generation.chain import answer_question

st.set_page_config(page_title="Smriti", page_icon="🧠")
st.title("🧠 Smriti — RAG Chatbot")


def build_retriever():
    """Rebuild hybrid + reranked retriever. Cheap now that the cross-encoder
    model itself is cached at module level in reranker.py — this just rebuilds
    the BM25 index over current Chroma contents."""
    hybrid = get_hybrid_retriever(st.session_state.vectorstore, k=25)
    return get_reranked_retriever(hybrid, top_n=3)


def render_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            conf = s.get("confidence")
            conf_str = f" (confidence: {conf:.2f})" if conf is not None else ""
            st.caption(f"{s['source']} — page {s.get('page', 'N/A')}{conf_str}")


# --- Session state setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-session"
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = get_vectorstore()
if "retriever" not in st.session_state:
    st.session_state.retriever = build_retriever()

# --- Document upload ---
st.subheader("Upload a document")
uploaded_file = st.file_uploader("PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    save_path = Path("tests/test_files") / uploaded_file.name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document..."):
        docs = load_document(str(save_path))
        chunks = chunk_documents(docs)
        add_documents(chunks)
        st.session_state.retriever = build_retriever()  # refresh BM25 index

    st.success(f"Ingested {len(chunks)} chunk(s) from {uploaded_file.name}")

st.divider()

# --- Document dashboard ---
st.subheader("📚 Document Dashboard")
docs_in_store = get_ingested_documents(st.session_state.vectorstore)

if docs_in_store:
    for d in docs_in_store:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"**{d['source']}** — {d['chunks']} chunk(s)")
        with col2:
            if st.button("🗑️ Delete", key=f"delete_{d['source']}"):
                deleted = delete_document(st.session_state.vectorstore, d["source"])
                st.session_state.retriever = build_retriever()
                st.success(f"Deleted {deleted} chunk(s) from {d['source']}")
                st.rerun()

    st.divider()

    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False

    if not st.session_state.confirm_clear:
        if st.button("🧹 Clear all documents"):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("This will permanently delete ALL documents. Are you sure?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, delete everything"):
                clear_all_documents(st.session_state.vectorstore)
                st.session_state.vectorstore = get_vectorstore()  # reconnect to fresh empty collection
                st.session_state.retriever = build_retriever()
                st.session_state.confirm_clear = False
                st.success("All documents cleared.")
                st.rerun()
        with c2:
            if st.button("Cancel"):
                st.session_state.confirm_clear = False
                st.rerun()
else:
    st.caption("No documents ingested yet.")