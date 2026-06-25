from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

GENERATION_MODEL = "qwen3:8b"

SYSTEM_PROMPT = """/no_think
You are Smriti, a helpful assistant that answers questions using only 
the provided context. If the answer isn't in the context, say you 
don't know — do not make up information.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

def get_llm():
    return ChatOllama(model=GENERATION_MODEL, temperature=0)

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain():
    return prompt | get_llm() | StrOutputParser()

def answer_question(question: str, retriever) -> dict:
    docs = retriever.invoke(question)
    context = format_docs(docs)

    chain = build_chain()
    answer = chain.invoke({"context": context, "question": question})

    sources = [
        {"source": d.metadata.get("source"), "page": d.metadata.get("page")}
        for d in docs
    ]
    return {"answer": answer, "sources": sources}