import json
from pathlib import Path

import pandas as pd
import streamlit as st


REPORT_DIR = Path("evaluation/reports")


st.set_page_config(
    page_title="Enterprise Knowledge Engine",
    page_icon="",
    layout="wide",
)


def latest_file(pattern: str) -> Path | None:
    files = sorted(REPORT_DIR.glob(pattern))
    return files[-1] if files else None


@st.cache_data
def load_reports(summary_path: str, results_path: str) -> tuple[dict, pd.DataFrame]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    results = pd.read_csv(results_path)
    return summary, results


def format_rate(value: float) -> str:
    return f"{value * 100:.0f}%"


def status_label(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


summary_path = latest_file("eval_summary_*.json")
results_path = latest_file("eval_results_*.csv")

st.title("Enterprise Knowledge Engine")
st.caption("RAG evaluation dashboard")

if not summary_path or not results_path:
    st.warning("No evaluation reports found. Run `python -m evaluation.metrics` first.")
    st.stop()

summary, results = load_reports(str(summary_path), str(results_path))
results["status"] = results["passed"].map(status_label)

with st.sidebar:
    st.header("Report")
    st.write(summary_path.name)
    st.write(results_path.name)
    st.divider()
    st.write("Run a fresh evaluation:")
    st.code("python -m evaluation.metrics", language="bash")

metric_cols = st.columns(5)
metric_cols[0].metric("Pass Rate", format_rate(summary["pass_rate"]))
metric_cols[1].metric("Citation Valid", format_rate(summary["citation_valid_rate"]))
metric_cols[2].metric("Source Hit", format_rate(summary["source_hit_rate"]))
metric_cols[3].metric("Term Recall", format_rate(summary["avg_term_recall"]))
metric_cols[4].metric("Avg Latency", f"{summary['avg_latency_seconds']:.2f}s")

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Evaluation Cases")
    display_cols = [
        "status",
        "id",
        "citation_valid",
        "source_hit",
        "term_recall",
        "latency_ok",
        "total_latency_seconds",
        "generation_latency_seconds",
    ]
    st.dataframe(
        results[display_cols],
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.subheader("Latency")
    latency_df = results[[
        "id",
        "total_latency_seconds",
        "generation_latency_seconds",
    ]].set_index("id")
    st.bar_chart(latency_df)

st.divider()

quality_cols = st.columns(3)
with quality_cols[0]:
    st.subheader("Pass / Fail")
    st.bar_chart(results["status"].value_counts())

with quality_cols[1]:
    st.subheader("Citation Validity")
    st.bar_chart(results["citation_valid"].map({True: "valid", False: "invalid"}).value_counts())

with quality_cols[2]:
    st.subheader("Source Hits")
    st.bar_chart(results["source_hit"].map({True: "hit", False: "miss"}).value_counts())

st.divider()

st.subheader("Case Details")
selected_id = st.selectbox("Evaluation case", results["id"].tolist())
case = results[results["id"] == selected_id].iloc[0]

detail_cols = st.columns(4)
detail_cols[0].metric("Status", case["status"])
detail_cols[1].metric("Term Recall", f"{case['term_recall']:.2f}")
detail_cols[2].metric("Total Latency", f"{case['total_latency_seconds']:.2f}s")
detail_cols[3].metric("Generation Latency", f"{case['generation_latency_seconds']:.2f}s")

st.write("Question")
st.info(case["question"])

st.write("Returned Sources")
sources = str(case["returned_sources"]).split(",")
st.write(pd.DataFrame({"source": sources}))

st.write("Answer Preview")
st.write(case["answer_preview"])
