import os
from functools import lru_cache
from typing import Literal

import cohere
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from retrieval.hybrid import query_hybrid

load_dotenv()

COHERE_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
DEFAULT_HYBRID_K = 20
DEFAULT_RERANK_K = 5

RerankProvider = Literal["cohere", "cross_encoder"]


def _chunk_to_document(chunk: dict) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source", "unknown")
    page = metadata.get("page", "unknown")
    return f"Source: {source}\nPage: {page}\n\n{chunk['text']}"


@lru_cache(maxsize=1)
def get_cohere_client():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY is not set")

    return cohere.Client(api_key)


@lru_cache(maxsize=1)
def get_cross_encoder(model_name: str = CROSS_ENCODER_MODEL):
    return CrossEncoder(model_name)


def rerank_with_cohere(
    query: str,
    chunks: list[dict],
    top_k: int = DEFAULT_RERANK_K,
    model: str = COHERE_MODEL
) -> list[dict]:
    documents = [_chunk_to_document(chunk) for chunk in chunks]
    client = get_cohere_client()

    response = client.rerank(
        query=query,
        documents=documents,
        model=model,
        top_n=min(top_k, len(chunks)),
        return_documents=False
    )

    reranked = []
    for rank, result in enumerate(response.results):
        chunk = chunks[result.index].copy()
        chunk["rerank_score"] = result.relevance_score
        chunk["rerank_rank"] = rank
        chunk["rerank_provider"] = "cohere"
        reranked.append(chunk)

    return reranked


def rerank_with_cross_encoder(
    query: str,
    chunks: list[dict],
    top_k: int = DEFAULT_RERANK_K,
    model_name: str = CROSS_ENCODER_MODEL
) -> list[dict]:
    model = get_cross_encoder(model_name)
    pairs = [(query, _chunk_to_document(chunk)) for chunk in chunks]
    scores = model.predict(pairs)

    ranked = sorted(
        enumerate(scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    reranked = []
    for rank, (chunk_index, score) in enumerate(ranked[:top_k]):
        chunk = chunks[chunk_index].copy()
        chunk["rerank_score"] = float(score)
        chunk["rerank_rank"] = rank
        chunk["rerank_provider"] = "cross_encoder"
        reranked.append(chunk)

    return reranked


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = DEFAULT_RERANK_K,
    provider: RerankProvider = "cohere"
) -> list[dict]:
    """
    Rerank retrieved chunks against the query and return the most relevant ones.
    """
    if not chunks:
        return []

    if provider == "cohere":
        return rerank_with_cohere(query, chunks, top_k)

    if provider == "cross_encoder":
        return rerank_with_cross_encoder(query, chunks, top_k)

    raise ValueError(f"Unsupported rerank provider: {provider}")


def retrieve_and_rerank(
    query: str,
    hybrid_k: int = DEFAULT_HYBRID_K,
    top_k: int = DEFAULT_RERANK_K,
    provider: RerankProvider = "cohere"
) -> list[dict]:
    """
    Convenience wrapper for the full retrieval stage:
    hybrid retrieval top N -> reranker top K.
    """
    hybrid_results = query_hybrid(
        query=query,
        dense_k=hybrid_k,
        sparse_k=hybrid_k,
        final_k=hybrid_k
    )

    return rerank_chunks(
        query=query,
        chunks=hybrid_results,
        top_k=top_k,
        provider=provider
    )
