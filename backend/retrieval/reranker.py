import operator
import math
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    return _cross_encoder


class ScoringCrossEncoderReranker(CrossEncoderReranker):
    """Same as CrossEncoderReranker, but keeps the relevance score
    by attaching it to each document's metadata instead of discarding it."""

    def compress_documents(self, documents, query, callbacks=None):
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        ranked = sorted(zip(documents, scores), key=operator.itemgetter(1), reverse=True)

        top = []
        for doc, raw_score in ranked[: self.top_n]:
            # Cross-encoder scores are raw logits, not probabilities.
            # Sigmoid maps them to a 0-1 range for a readable "confidence" display —
            # this is an approximation for UI purposes, not a calibrated probability.
            confidence = 1 / (1 + math.exp(-raw_score))
            doc.metadata = {**doc.metadata, "relevance_score": round(float(confidence), 3)}
            top.append(doc)
        return top


def get_reranked_retriever(base_retriever, top_n: int = 3):
    compressor = ScoringCrossEncoderReranker(model=get_cross_encoder(), top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    ) 