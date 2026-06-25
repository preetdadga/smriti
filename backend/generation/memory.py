from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# In-process only — lost on restart. Swap for a persistent store later if needed.
_session_store: dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]

def wrap_with_memory(chain):
    return RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )