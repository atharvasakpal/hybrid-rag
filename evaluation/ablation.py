import json
import time
from pathlib import Path

import pandas as pd

from evaluation.eval_dataset import get_eval_dataset
from evaluation.metrics import has_expected_source, expected_term_recall
from retrieval.dense import query_chroma
from retrieval.sparse import query_elasticsearch
from retrieval.hybrid import query_hybrid
from reranking.reranker import rerank_chunks
from generation.generator import generate_answer

REPORT_DIR = Path("evaluation/reports")
TOP_K = 5
SLEEP_BETWEEN_CALLS_SECONDS = 8.0  # stay under Groq free-tier 6000 TPM limit


def get_chunks_for_config(query: str, config: str) -> list[dict]:
    """
    Return top-5 chunks for a given retrieval configuration.
    All configs end with the same number of chunks so generation
    quality differences come from retrieval quality, not chunk count.
    """
    if config == "dense_only":
        return query_chroma(query, n_results=TOP_K)

    if config == "sparse_only":
        return query_elasticsearch(query, n_results=TOP_K)

    if config == "hybrid_rrf":
        return query_hybrid(query, dense_k=20, sparse_k=20, final_k=TOP_K)

    if config == "hybrid_rerank":
        hybrid_results = query_hybrid(query, dense_k=20, sparse_k=20, final_k=20)
        return rerank_chunks(query, hybrid_results, top_k=TOP_K, provider="cohere")

    raise ValueError(f"Unknown config: {config}")


def run_ablation(dataset: list[dict] | None = None) -> pd.DataFrame:
    dataset = dataset or get_eval_dataset()
    configs = ["dense_only", "sparse_only", "hybrid_rrf", "hybrid_rerank"]

    rows = []
    total_calls = len(dataset) * len(configs)
    call_index = 0

    for case in dataset:
        query = case["question"]
        print(f"\nQuestion: {query}")

        for config in configs:
            call_index += 1
            start = time.time()
            chunks = get_chunks_for_config(query, config)
            retrieval_time = time.time() - start

            result = generate_answer(query, chunks)

            source_hit = has_expected_source(result, case.get("expected_sources", []))
            term_recall = expected_term_recall(
                result.get("answer", ""),
                case.get("expected_terms", [])
            )

            rows.append({
                "question_id": case["id"],
                "config": config,
                "source_hit": source_hit,
                "term_recall": round(term_recall, 3),
                "citation_valid": result.get("citation_validation", {}).get("valid", False),
                "retrieval_time_seconds": round(retrieval_time, 3),
                "num_chunks_retrieved": len(chunks),
            })

            print(f"  {config:20} | source_hit={source_hit} | term_recall={term_recall:.2f} | retrieval={retrieval_time:.2f}s")

            # stay under Groq free-tier tokens-per-minute limit between calls
            if call_index < total_calls:
                time.sleep(SLEEP_BETWEEN_CALLS_SECONDS)

    return pd.DataFrame(rows)


def summarize_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-config averages across all questions.
    This is the table that goes in your README and LinkedIn post.
    """
    summary = df.groupby("config").agg(
        source_hit_rate=("source_hit", "mean"),
        avg_term_recall=("term_recall", "mean"),
        citation_valid_rate=("citation_valid", "mean"),
        avg_retrieval_time_seconds=("retrieval_time_seconds", "mean"),
    ).round(3)

    # order configs logically instead of alphabetically
    config_order = ["dense_only", "sparse_only", "hybrid_rrf", "hybrid_rerank"]
    summary = summary.reindex(config_order)

    return summary


def save_ablation_report(df: pd.DataFrame, summary: pd.DataFrame) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    detail_path = REPORT_DIR / f"ablation_detail_{timestamp}.csv"
    summary_path = REPORT_DIR / f"ablation_summary_{timestamp}.csv"

    df.to_csv(detail_path, index=False)
    summary.to_csv(summary_path)

    return detail_path, summary_path


def main():
    df = run_ablation()
    summary = summarize_ablation(df)

    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(summary.to_string())

    detail_path, summary_path = save_ablation_report(df, summary)
    print(f"\nSaved detail: {detail_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()