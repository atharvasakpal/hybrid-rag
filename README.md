# hybrid-rag

A production-style RAG system built to go beyond the basic "load PDF, embed, ask a question" tutorial — hybrid retrieval, reranking, grounded citations, automated evaluation, and a scale test that surfaced and fixed two real bugs.

Built on a medical document corpus (clinical case reports, treatment guidelines, infectious disease research) because precise terminology makes the gap between retrieval strategies measurable, not just theoretical.

---

## Proof, not claims

- **61,914 chunks indexed** across 1,999 real medical documents (PubMed Central open-access corpus) — 133x document scale, 46x chunk scale versus the initial 15-document corpus
- **Hybrid + reranking improves answer term recall by +11.8%** over dense-only retrieval (0.844 → 0.944), measured via a 15-question evaluation set with every fact manually verified against source text — no invented ground truth
- **Retrieval stays fast at scale**: dense ~0.21s, sparse ~0.03s, hybrid ~0.08s at 61,914 chunks (post model warm-up)
- **Two real bugs found and fixed via scale testing**, invisible at small scale: a file-descriptor leak that crashed ingestion at ~1,148 documents, and an Elasticsearch pagination cap that silently truncated deduplication checks above 10,000 results
- **Citation validity held at 93–100%** across all evaluation runs, enforced via prompt instructions plus a regex-based validator with automatic retry on failure

---

## Architecture

```
Ingestion        → Multi-format loader (PDF / DOCX / TXT / PMC XML)
                   → Semantic chunking (topic-boundary splitting, not fixed token size)
                   → Dual storage: ChromaDB (vectors) + Elasticsearch (BM25)

Hybrid Retrieval → Dense search (ChromaDB) + Sparse search (Elasticsearch BM25)
                   → Merged via Reciprocal Rank Fusion (RRF, k=60)

Reranking        → Top 20 RRF candidates → Cohere Rerank v3.5
                   (local cross-encoder also supported, used in ablation)
                   → Top 5 returned

Generation       → Top 5 chunks → Gemini 2.5 Flash Lite
                   → Answer with [S1]-style citations
                   → Regex validator → auto-retry on invalid citations

Evaluation       → 15-question verified eval set
                   → Citation validity, source hit rate, term recall, latency
                   → 4-way ablation comparison across retrieval configs
                   → Streamlit dashboard for results
```

---

## The ablation finding

Four retrieval configurations run against the same 15-question evaluation set, scored identically:

| Config | Source Hit Rate | Avg Term Recall |
|---|---|---|
| Dense only | 1.0 | 0.844 |
| Sparse only | 1.0 | 0.900 |
| Hybrid (RRF) | 1.0 | 0.900 |
| **Hybrid + Rerank** | **1.0** | **0.944** |

The clearest single example: asked for the exact incidence rate of euglycemic diabetic ketoacidosis, **dense-only retrieval missed the figure entirely (term recall 0.00)** — semantic similarity did not surface the specific numeric chunk. **Sparse (BM25) retrieval found it immediately (term recall 1.00)** via exact keyword match. This is the precise failure mode hybrid retrieval exists to solve, demonstrated empirically rather than asserted.

At scale (61,914 chunks), the same dynamic appeared again: on one query, sparse retrieval alone was pulled toward a different document in the larger corpus, while dense retrieval stayed correct — and hybrid RRF fusion recovered the correct source as the top result.

---

## What scale testing found

Ingesting 1,999 real documents surfaced two bugs invisible at 15 or 50 document scale:

**Bug 1 — File descriptor leak**

`get_chroma_client()` and `get_es_client()` were not cached, creating a new HTTP client per document instead of once per pipeline run. After ~1,148 documents in a single process, this exhausted the OS file descriptor limit. Fixed with `@lru_cache(maxsize=1)`, matching the pattern already used correctly elsewhere in the codebase (`get_groq_client`, `get_cohere_client`).

Because the pipeline was built with per-document fault tolerance and chunk-ID-based deduplication from day one, the run resumed cleanly with zero data loss — only the ~850 failed documents were reprocessed, the ~1,100 already completed were skipped.

**Bug 2 — Elasticsearch pagination cap**

The deduplication check used a flat search with `size=10000`, which silently truncates above Elasticsearch's default result window. Past 10,000 indexed chunks, the check undercounted existing chunks, causing redundant re-indexing. Because `chunk_id` is the Elasticsearch document `_id`, this was harmless overwrites, not data corruption. Verified via direct `_count` API. Fixed using Elasticsearch's scroll API for correct pagination at any scale.

Both findings documented in full in `DECISIONS.md`.

---

## Stack

| Layer | Technology |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` (local, sentence-transformers) |
| Vector store | ChromaDB (Docker) |
| Keyword index | Elasticsearch 8.11.0 (Docker) |
| Reranker | Cohere Rerank v3.5 + local cross-encoder |
| LLM | Google Gemini 2.5 Flash Lite |
| API | FastAPI + uvicorn |
| Dashboard | Streamlit |
| Infrastructure | Docker Compose |

---

## Running locally

**Prerequisites:** Docker running, Python 3.11+, API keys for Cohere and Google Gemini

```bash
# 1. Start databases
docker-compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Add GEMINI_API_KEY, COHERE_API_KEY to .env

# 4. Ingest documents (add your PDFs to data/papers/)
python -m ingestion.pipeline

# 5. Start the API
uvicorn api.main:app --reload

# 6. Run evaluation
python -m evaluation.metrics
python -m evaluation.ablation

# 7. View dashboard
streamlit run dashboard/app.py
```

**Example API call:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the relationship between SGLT2 inhibitors and diabetic ketoacidosis?",
    "hybrid_k": 20,
    "rerank_k": 5,
    "rerank_provider": "cohere"
  }'
```

---

## Known limitations

- **Chunk size variability in PMC corpus** — PMC XML articles saved as single text blobs occasionally produce large chunks. Mitigated by switching to Gemini (no hard TPM ceiling at this scale). Fix at source: split by `<sec>` XML tags before chunking.
- **Eval set is 15 questions** — sufficient to show retrieval differentiation between configs, not a comprehensive benchmark.
- **Latency on free-tier LLM** — Groq free tier has a 6,000 TPM ceiling that caused per-request failures at scale. Root-caused via direct 429 error inspection. Switched to Gemini free tier which has no equivalent constraint at this usage level.
- **No fairness or adversarial testing** — essential before any real clinical deployment, not done here.
- **Streamlit dashboard reads static eval reports** — not a live query interface. The FastAPI endpoint handles live queries.

---

## Engineering decisions

Full decision log including the latency investigation root cause, scale test findings, ablation methodology, and explicit corrections of initial wrong conclusions: [`DECISIONS.md`](./DECISIONS.md)
