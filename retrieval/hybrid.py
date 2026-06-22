from retrieval.dense import query_chroma
from retrieval.sparse import query_elasticsearch


RRF_K = 60


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    rrf_k: int = RRF_K
) -> list[dict]:
    """
    Merge dense and sparse retrieval results using Reciprocal Rank Fusion.

    RRF score = sum(1 / (rrf_k + rank + 1))
    rank is zero-based in dense/sparse retriever outputs.
    """
    fused: dict[str, dict] = {}

    for retriever_name, results in (
        ("dense", dense_results),
        ("sparse", sparse_results),
    ):
        for rank, result in enumerate(results):
            chunk_id = result["chunk_id"]
            rrf_score = 1 / (rrf_k + rank + 1)

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                    "dense_score": None,
                    "sparse_score": None,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "retrievers": [],
                }

            fused_result = fused[chunk_id]
            fused_result["rrf_score"] += rrf_score
            fused_result["retrievers"].append(retriever_name)

            if retriever_name == "dense":
                fused_result["dense_score"] = result["score"]
                fused_result["dense_rank"] = rank
            else:
                fused_result["sparse_score"] = result["score"]
                fused_result["sparse_rank"] = rank

    return sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )


def query_hybrid(
    query: str,
    dense_k: int = 20,
    sparse_k: int = 20,
    final_k: int = 10,
    rrf_k: int = RRF_K
) -> list[dict]:
    """
    Run hybrid retrieval:
    1. Dense vector search from ChromaDB.
    2. Sparse BM25 search from Elasticsearch.
    3. Reciprocal Rank Fusion to merge and deduplicate.
    """
    dense_results = query_chroma(query, n_results=dense_k)
    sparse_results = query_elasticsearch(query, n_results=sparse_k)

    fused_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        rrf_k=rrf_k
    )

    return fused_results[:final_k]
