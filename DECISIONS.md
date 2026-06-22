# Engineering Decisions — hybrid-rag

## Why this project
Built to cover gaps versus 13+ JDs for 12-15 LPA AI/ML roles at companies like
Razorpay, Goldman Sachs, Mastercard, and Siemens. Target gaps: hybrid retrieval,
reranking, evaluation with real metrics, and production monitoring. Goal was a
system with measured tradeoffs and honest documentation, not a tutorial demo.

---

## Domain and data
Medical domain chosen over generic NLP papers because terminology density makes
hybrid retrieval's value measurable — exact drug names and clinical figures favor
BM25, conceptual questions favor dense embeddings. The contrast between the two
retrieval strategies is visible in real evaluation results, not just asserted.

Initial corpus: 15 PDFs sourced from PubMed Central and direct journal downloads,
manually verified. 256 pages extracted, 1,330 semantic chunks.

Scale corpus: 1,999 open-access articles from PMC E-Utilities API (diabetes
search, OA filter), saved as plain text. Final corpus: 1,999 documents,
61,914 chunks across both ChromaDB and Elasticsearch. 133x document scale,
46x chunk scale versus the initial corpus.

---

## Stack decisions

### Embedding model
all-MiniLM-L6-v2 via sentence-transformers — free, runs locally, 384-dimensional
vectors, cosine similarity. No API cost means no rate limits during development,
evaluation runs, or scale testing.

### LLM
Started with Groq free tier (llama-3.1-8b-instant). Switched to Google Gemini
(gemini-2.5-flash-lite) when Groq's Developer tier upgrade was temporarily
unavailable and the free-tier 6,000 TPM ceiling caused single-query failures
at the scaled-up corpus size (prompt with 5 large PMC chunks exceeded 6,000
tokens in one request). Gemini's free tier has no equivalent TPM constraint
at this usage level.

### Reranker
Cohere Rerank v3.5 — industry standard, used in production RAG systems at scale.
Free trial tier sufficient for this project. Architecture also supports a local
cross-encoder (ms-marco-MiniLM-L-6-v2) as a second provider behind the same
interface, used in ablation comparison.

### Vector store
ChromaDB — stores chunks as 384-dimensional embeddings for dense semantic search.
Chosen over Pinecone because it runs locally via Docker, no API cost, persistent
volume means data survives container restarts. cosine similarity space configured
explicitly, correct for normalized MiniLM vectors.

### Keyword index
Elasticsearch 8.11.0 — BM25 keyword search for sparse retrieval. Runs locally
via Docker. Dense retrieval alone misses exact medical terminology; sparse
retrieval alone misses semantic meaning. Both are needed. Single shard, zero
replicas — development config, not scale-tested for write throughput.

---

## Ingestion pipeline (Layer 1)

- Per-page PDF extraction so citations reference exact page numbers
- DOCX treated as single page — no native page structure in the format
- Pages under 50 characters skipped — handles scanned/empty pages
- SemanticChunker over fixed-size splitting — cuts on topic boundaries not token
  counts. breakpoint_threshold_amount=85, ~7 chunks per page on medical PDFs
- chunk_id = deterministic MD5 hash of source+page+chunk_index. Same chunk
  always gets the same ID across ChromaDB and Elasticsearch — required for RRF
  deduplication. Safe to re-run ingestion without creating duplicates.
- Both databases deduplicate on chunk_id before writing

### PMC bulk fetcher
Added ingestion/pmc_fetcher.py for the scale test corpus. Uses E-Utilities
esearch + efetch APIs (JATS XML format). Batches of 100 IDs, 1s sleep between
batches to respect E-Utilities rate limits (~3 req/s without API key). Each
article saved as a plain .txt file, processed by the existing pipeline.py
without modification — no new loader path needed.

---

## Retrieval (Layer 2)
Hybrid retrieval merges ChromaDB (dense/semantic) and Elasticsearch (sparse/BM25)
using Reciprocal Rank Fusion. rrf_k=60 (standard default from original RRF paper,
not yet tuned for this corpus). RRF rewards chunks found by both retrievers
without requiring agreement — a chunk found by only one retriever still ranks,
just without the cross-retriever bonus.

---

## Reranking (Layer 3)
Top 20 RRF-merged candidates reranked by Cohere Rerank v3.5 down to top 5.
Architecture also supports a local cross-encoder as a second provider behind
the same interface, used for ablation comparison. Provider is a runtime
parameter — no code change needed to switch.

---

## Generation (Layer 4)
Gemini generates answers strictly from the top 5 reranked chunks. Citations
enforced via prompt instructions requiring [S1]-style labels. A regex-based
validator checks every citation against the actual available labels. If
validation fails, one automatic retry is triggered with a corrective message.
Citation validity held at 93-100% across all evaluation runs.

System prompt and user prompt both explicitly forbid numeric paper references
like [3] or [15] — these appear in scientific paper text and the LLM would
otherwise copy them verbatim, producing citations that look real but reference
nothing in the retrieved context.

---

## Evaluation (Layer 5)

### Eval dataset construction
15 questions, expanded from an initial 5. Every question and its expected_terms
were verified against actual extracted PDF text — no fact was guessed or inferred
from titles or abstracts. Categories:
- Exact-term/numeric questions (favor BM25, dense may miss)
- Conceptual/paraphrased questions (favor dense, sparse may miss exact words)
- Disambiguation among similar-topic documents
- Cross-document questions requiring both source documents to answer

### Ablation study
Compared four retrieval configs (dense-only, sparse-only, hybrid RRF,
hybrid+rerank) feeding into identical generation logic, scored with the same
metrics as the main eval.

Results:
  dense_only:     source_hit_rate=1.0, avg_term_recall=0.844
  sparse_only:    source_hit_rate=1.0, avg_term_recall=0.900
  hybrid_rrf:     source_hit_rate=1.0, avg_term_recall=0.900
  hybrid_rerank:  source_hit_rate=1.0, avg_term_recall=0.944

Concrete proof point: question "What is the incidence rate of euglycemic
diabetic ketoacidosis" — dense-only term_recall=0.00 (missed the exact numeric
figure entirely), sparse-only term_recall=1.00 (found it via exact keyword
match). This is the precise failure mode hybrid retrieval exists to fix,
demonstrated empirically rather than asserted.

Source hit rate was 1.0 across all four configs — the corpus is not large enough
or ambiguous enough to differentiate configs on source-finding. Only term recall
shows real differentiation at this corpus size.

---

## Latency investigation — full root cause history

### Phase 1: Initial observation
Identical queries showed wildly inconsistent latency (0.5s to 20s+). Ruled out
via direct isolated testing: Groq rate limiting (headers confirmed healthy quota
at the time), citation retry loop (not triggered on slow queries), prompt size
(normal ~2000 tokens), content of specific queries (same query fast one run,
slow another).

First (incorrect) conclusion: assumed transient Groq infrastructure noise, since
raw test calls with max_tokens=20 never reproduced the slowness.

### Phase 2: Root cause found
Direct 429 error during ablation run revealed the real cause: Groq free tier
enforces a 6,000 tokens-per-minute (TPM) limit. Sequential eval/ablation calls
at ~2,000-2,700 tokens each exceeded this within a rolling 60-second window.
Groq's behavior under that ceiling is inconsistent — sometimes silent queuing
(manifests as 15-40s delays), sometimes a hard 429 failure.

Earlier raw test with max_tokens=20 never reproduced it because trivially small
requests don't approach the real token usage of full RAG prompts.

Fix: 8-second sleep between Groq calls in evaluation/ablation scripts only
(not in the production API path, where single-query usage doesn't approach
the TPM ceiling). Pass rate improved from 0.2 to 0.533.

### Phase 3: Provider switch
At scale (61,914 chunks), the top 5 reranked chunks from PMC articles were
significantly larger than original PDF chunks (9,541 chars vs ~300-500 chars).
A single query prompt exceeded 6,000 tokens, making the free-tier TPM limit
a per-request blocker, not just a sequential-call accumulation problem.

Switched to Google Gemini (gemini-2.5-flash-lite) — no equivalent TPM constraint
at this usage level on the free tier. Single-function swap in generator.py,
no other files changed.

### Correction note
The first conclusion (infra noise) was wrong and was explicitly corrected once
the real cause was found, rather than left in the record. Lesson: don't conclude
"infrastructure noise" without checking for explicit rate-limit errors first.

---

## Retrieval cold-start latency (resolved)
One anomalous 98s and repeated ~10s retrieval times observed in ablation runs,
isolated to query_chroma. Root cause: HuggingFace embedding model loads into
memory on first call per Python process (~9-10s one-time cost). All subsequent
calls in the same process are under 100ms.

Production impact: none. In api/main.py, the model loads once at server startup,
not per request. This cold-start only appears in standalone test scripts because
each python -c invocation starts a fresh process.

---

## Scale test results (1,999 documents, PMC open-access corpus)
Corpus grew from 15 docs/1,330 chunks to 1,999 docs/61,914 chunks. Two real
bugs surfaced only at this scale:

### Bug 1: File descriptor leak
get_chroma_client() and get_es_client() were not cached with @lru_cache,
creating a new HTTP client (and underlying socket) per document instead of once
per pipeline run. After ~1,148 documents in a single process, this exhausted
the OS file descriptor limit. Every subsequent document failed.

Fixed with @lru_cache(maxsize=1), matching the pattern already used correctly
elsewhere (get_groq_client, get_cohere_client, get_embeddings_model). Zero data
loss on resume — the pipeline was built with per-document fault tolerance and
chunk_id-based deduplication from day one, so the run resumed cleanly and
processed only the ~850 documents that had failed, skipping the ~1,100 already
completed.

This bug was invisible at 15 and 50 document scale. Only appeared above ~1,000
documents in a single process — a textbook example of why scale testing matters
beyond functional correctness testing.

### Bug 2: Elasticsearch dedup check capped at 10,000 results
The deduplication query in index_chunks used a flat search with size=10000,
which silently truncates at Elasticsearch's default max result window. Past
10,000 indexed chunks, the dedup check undercounted what actually existed,
causing redundant re-indexing. Because chunk_id is the Elasticsearch document
_id, redundant indexing is just an overwrite — no data corruption.

Verified via direct _count API that actual stored data was correct throughout
(61,914 in both ChromaDB and Elasticsearch at completion). Fixed using
Elasticsearch's scroll API for safe pagination past the 10K window.

---

## Retrieval latency and correctness at scale (61,914 chunks)
Measured after scaling corpus 46x (1,330 → 61,914 chunks). Excluding one-time
embedding model cold start (~10s per process start, not a scale effect):

  Dense (ChromaDB):   ~0.21s
  Sparse (ES BM25):   ~0.03s
  Hybrid (RRF):       ~0.08s

All three remain well within interactive latency at 46x original scale.

Correctness: 3 original eval queries re-run against the full scaled corpus.
2/3: all methods correctly found the original target document despite 60,000+
competing unrelated chunks. 1/3 (SGLT2/DKA): sparse retrieval alone was pulled
off course by a new, more relevant PMC paper — which is actually correct behavior,
not a failure. The new paper (PMC13265897.txt) is a better source for this query
than the original case report. Hybrid RRF correctly surfaced it as the top result.

---

## Chunk size problem discovered at scale
PMC XML articles were saved as a single text blob per article in pmc_fetcher.py,
with no section/page boundaries. Semantic chunker occasionally produced very
large chunks (up to 9,541 chars) from long topically-consistent sections.
These oversized chunks caused the single-query TPM failure described above.

Known limitation: existing 61,914 chunks already stored have this variability.
Fix for future ingestion: either split PMC articles by <sec> XML tags before
saving (correct fix), or add a hard character ceiling in chunker.py as a safety
net (defensive fix). Not re-run at scale due to time cost.

---

## Known limitations

1. Chunk size variability in PMC corpus — some chunks significantly larger than
   PDF-derived chunks. Affects prompt size. Mitigated by switching to Gemini
   (no hard TPM ceiling at this scale), not fixed at source.

2. Elasticsearch dedup uses scroll API for correctness but is slower than the
   previous flat search on small corpora. Acceptable tradeoff.

3. Eval set is 15 questions — sufficient to show retrieval differentiation, not
   a comprehensive benchmark.

4. No adversarial robustness testing on generated answers.

5. No fairness evaluation across demographic subgroups — essential before any
   real clinical deployment.

6. Single-shard, zero-replica Elasticsearch — development config. Production
   would use multi-shard with replication for fault tolerance.

7. Streamlit dashboard reads static eval report files, not live query results.
   A live query interface would require embedding the FastAPI call inside
   Streamlit, which is a straightforward addition not yet built.

---

## What this project supports claiming
- A working hybrid retrieval + reranking + grounded citation RAG pipeline
- An empirically measured (+11.8% term recall) improvement from hybrid+rerank
  over dense-only, with one concrete reproducible example (EDKA incidence rate
  query: dense=0.00, sparse=1.00 term recall)
- A scale test at 1,999 documents / 61,914 chunks that found and fixed two
  real production bugs invisible at small scale
- Sub-250ms retrieval latency at 46x original corpus scale
- A fully documented, root-caused latency investigation with explicit correction
  of an initial wrong conclusion

## What this project does not support claiming
- Scales to 10,000+ documents (tested at ~2,000; extrapolation is not evidence)
- Production-ready for clinical use (no fairness eval, no adversarial testing)
- Any specific latency SLA (free-tier provider behavior documented as variable)
