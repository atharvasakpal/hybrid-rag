import json
import time
from pathlib import Path

import pandas as pd

from evaluation.eval_dataset import get_eval_dataset
from generation.generator import answer_query

DEFAULT_LATENCY_THRESHOLD_SECONDS = 15.0
REPORT_DIR = Path("evaluation/reports")
SLEEP_BETWEEN_CALLS_SECONDS = 8.0  # stay under Groq free-tier 6000 TPM limit


def _normalize(text: str) -> str:
    return text.lower()


def has_expected_source(result: dict, expected_sources: list[str]) -> bool:
    returned_sources = {
        citation["source"]
        for citation in result.get("citations", [])
    }
    return any(source in returned_sources for source in expected_sources)


def expected_term_recall(answer: str, expected_terms: list[str]) -> float:
    if not expected_terms:
        return 1.0

    normalized_answer = _normalize(answer)
    matched_terms = [
        term for term in expected_terms
        if _normalize(term) in normalized_answer
    ]
    return len(matched_terms) / len(expected_terms)


def evaluate_result(
    case: dict,
    result: dict,
    latency_threshold_seconds: float = DEFAULT_LATENCY_THRESHOLD_SECONDS
) -> dict:
    citation_valid = result.get("citation_validation", {}).get("valid", False)
    source_hit = has_expected_source(result, case.get("expected_sources", []))
    term_recall = expected_term_recall(
        result.get("answer", ""),
        case.get("expected_terms", [])
    )
    total_latency = result.get("total_latency_seconds", 0.0)
    latency_ok = total_latency <= latency_threshold_seconds

    passed = (
        bool(result.get("answer"))
        and citation_valid
        and source_hit
        and term_recall >= 0.6
        and latency_ok
    )

    return {
        "id": case["id"],
        "question": case["question"],
        "passed": passed,
        "citation_valid": citation_valid,
        "source_hit": source_hit,
        "term_recall": round(term_recall, 3),
        "latency_ok": latency_ok,
        "total_latency_seconds": round(total_latency, 3),
        "generation_latency_seconds": round(result.get("latency_seconds", 0.0), 3),
        "model": result.get("model"),
        "used_citations": ",".join(
            result.get("citation_validation", {}).get("used_labels", [])
        ),
        "returned_sources": ",".join(
            citation["source"]
            for citation in result.get("citations", [])
        ),
        "answer_preview": result.get("answer", "")[:300],
    }


def run_evaluation(
    dataset: list[dict] | None = None,
    hybrid_k: int = 20,
    rerank_k: int = 5,
    rerank_provider: str = "cohere",
    latency_threshold_seconds: float = DEFAULT_LATENCY_THRESHOLD_SECONDS
) -> list[dict]:
    dataset = dataset or get_eval_dataset()
    rows = []

    for index, case in enumerate(dataset, start=1):
        print(f"[{index}/{len(dataset)}] Evaluating: {case['id']}")
        start_time = time.time()

        try:
            result = answer_query(
                query=case["question"],
                hybrid_k=hybrid_k,
                rerank_k=rerank_k,
                rerank_provider=rerank_provider,
            )
            row = evaluate_result(case, result, latency_threshold_seconds)
        except Exception as exc:
            row = {
                "id": case["id"],
                "question": case["question"],
                "passed": False,
                "citation_valid": False,
                "source_hit": False,
                "term_recall": 0.0,
                "latency_ok": False,
                "total_latency_seconds": round(time.time() - start_time, 3),
                "generation_latency_seconds": 0.0,
                "model": None,
                "used_citations": "",
                "returned_sources": "",
                "answer_preview": "",
                "error": str(exc),
            }

        rows.append(row)
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"  {status} | citation={row['citation_valid']} "
            f"source={row['source_hit']} terms={row['term_recall']} "
            f"latency={row['total_latency_seconds']}s"
        )

        # stay under Groq free-tier tokens-per-minute limit between calls
        if index < len(dataset):
            time.sleep(SLEEP_BETWEEN_CALLS_SECONDS)

    return rows


def summarize_results(rows: list[dict]) -> dict:
    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])

    if total == 0:
        return {
            "total": 0,
            "passed": 0,
            "pass_rate": 0.0,
            "avg_latency_seconds": 0.0,
            "avg_term_recall": 0.0,
            "citation_valid_rate": 0.0,
            "source_hit_rate": 0.0,
        }

    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3),
        "avg_latency_seconds": round(
            sum(row["total_latency_seconds"] for row in rows) / total,
            3
        ),
        "avg_term_recall": round(
            sum(row["term_recall"] for row in rows) / total,
            3
        ),
        "citation_valid_rate": round(
            sum(1 for row in rows if row["citation_valid"]) / total,
            3
        ),
        "source_hit_rate": round(
            sum(1 for row in rows if row["source_hit"]) / total,
            3
        ),
    }


def save_report(rows: list[dict], summary: dict) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    csv_path = REPORT_DIR / f"eval_results_{timestamp}.csv"
    json_path = REPORT_DIR / f"eval_summary_{timestamp}.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return csv_path, json_path


def main():
    rows = run_evaluation()
    summary = summarize_results(rows)
    csv_path, json_path = save_report(rows, summary)

    print("\nEvaluation summary")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()