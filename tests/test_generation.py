from backend.retrieval.vectorstore import get_vectorstore
from backend.generation.chain import answer_question

store = get_vectorstore()
retriever = store.as_retriever(search_kwargs={"k": 3})

result = answer_question("What is the capital of France?", retriever)

print("--- Answer ---")
print(result["answer"])
print("\n--- Sources ---")
for s in result["sources"]:
    print(s)