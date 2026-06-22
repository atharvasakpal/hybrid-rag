import time
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from generation.generator import answer_query


app = FastAPI(
    title="Enterprise Knowledge Engine API",
    description="Production-style RAG API with hybrid retrieval, reranking, generation, and citation validation.",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User question to answer from the indexed knowledge base.")
    hybrid_k: int = Field(20, ge=1, le=100, description="Number of hybrid retrieval candidates before reranking.")
    rerank_k: int = Field(5, ge=1, le=20, description="Number of chunks to keep after reranking.")
    rerank_provider: Literal["cohere", "cross_encoder"] = Field(
        "cohere",
        description="Reranking backend to use."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is the relationship between SGLT2 inhibitors and diabetic ketoacidosis?",
                "hybrid_k": 20,
                "rerank_k": 5,
                "rerank_provider": "cohere",
            }
        }
    }


class Source(BaseModel):
    label: str
    source: str
    page: int | str
    chunk_id: str | None = None
    rerank_score: float | None = None
    rrf_score: float | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]
    citation_validation: dict[str, Any]
    model: str
    generation_latency_seconds: float
    total_latency_seconds: float


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


@app.get("/", response_model=HealthResponse)
def root():
    return {
        "status": "ok",
        "service": "enterprise-knowledge-engine",
        "version": app.version,
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "service": "enterprise-knowledge-engine",
        "version": app.version,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    start_time = time.time()

    try:
        result = answer_query(
            query=request.query,
            hybrid_k=request.hybrid_k,
            rerank_k=request.rerank_k,
            rerank_provider=request.rerank_provider,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {exc}") from exc

    sources = [
        Source(
            label=citation["label"],
            source=citation["source"],
            page=citation["page"],
            chunk_id=citation.get("chunk_id"),
            rerank_score=citation.get("rerank_score"),
            rrf_score=citation.get("rrf_score"),
        )
        for citation in result["citations"]
    ]

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        sources=sources,
        citation_validation=result["citation_validation"],
        model=result["model"],
        generation_latency_seconds=round(result["latency_seconds"], 4),
        total_latency_seconds=round(result.get("total_latency_seconds", time.time() - start_time), 4),
    )
