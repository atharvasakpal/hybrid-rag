import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from tqdm import tqdm

load_dotenv()

ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", 9200))
ES_INDEX = os.getenv("ES_INDEX_NAME", "rag_documents")
BATCH_SIZE = 100


from functools import lru_cache

@lru_cache(maxsize=1)
def get_es_client() -> Elasticsearch:
    return Elasticsearch(
        f"http://{ES_HOST}:{ES_PORT}",
        request_timeout=30
    )


def create_index_if_not_exists(es: Elasticsearch):
    """
    Create index with correct mappings.
    Safe to call multiple times — skips if already exists.
    """
    try:
        es.indices.get(index=ES_INDEX)
        print(f"Index '{ES_INDEX}' already exists")
        return
    except Exception:
        pass  # index doesn't exist, create it

    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "medical_analyzer": {
                        "type": "standard",
                        "stopwords": "_english_"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "text":        {"type": "text", "analyzer": "medical_analyzer"},
                "chunk_id":    {"type": "keyword"},
                "source":      {"type": "keyword"},
                "file_path":   {"type": "keyword"},
                "file_type":   {"type": "keyword"},
                "category":    {"type": "keyword"},
                "ingested_at": {"type": "date"},
                "page":        {"type": "integer"},
                "total_pages": {"type": "integer"},
                "chunk_index": {"type": "integer"},
                "word_count":  {"type": "integer"}
            }
        }
    }

    es.indices.create(index=ES_INDEX, body=mapping)
    print(f"✓ Created Elasticsearch index: {ES_INDEX}")


def index_chunks(chunks: list[dict]):
    """
    Store chunks in Elasticsearch for BM25 search.
    Uses chunk_id as document _id — Elasticsearch deduplicates automatically.
    """
    es = get_es_client()
    create_index_if_not_exists(es)

    # check existing
    existing = set()
    try:
        resp = es.search(
            index=ES_INDEX,
            body={
                "query": {"match_all": {}},
                "_source": ["chunk_id"],
                "size": 10000
            }
        )
        existing = {hit["_source"]["chunk_id"] for hit in resp["hits"]["hits"]}
        print(f"Found {len(existing)} existing chunks in Elasticsearch")
    except Exception:
        pass

    new_chunks = [c for c in chunks if c["chunk_id"] not in existing]
    print(f"New chunks to index: {len(new_chunks)}")

    if not new_chunks:
        print("Nothing to add — all chunks already in Elasticsearch")
        return

    for i in tqdm(range(0, len(new_chunks), BATCH_SIZE), desc="Indexing batches"):
        batch = new_chunks[i: i + BATCH_SIZE]
        operations = []

        for chunk in batch:
            operations.append({
                "index": {
                    "_index": ES_INDEX,
                    "_id": chunk["chunk_id"]
                }
            })
            operations.append({
                "text": chunk["text"],
                "chunk_id": chunk["chunk_id"],
                **chunk["metadata"]
            })

        es.bulk(operations=operations, refresh=True)

    total = es.count(index=ES_INDEX)["count"]
    print(f"✓ Elasticsearch now contains {total} chunks")


def query_elasticsearch(query: str, n_results: int = 20) -> list[dict]:
    """
    Sparse BM25 retrieval from Elasticsearch.
    Returns results in identical format to query_chroma for RRF compatibility.
    """
    es = get_es_client()

    resp = es.search(
        index=ES_INDEX,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text"],
                    "type": "best_fields"
                }
            },
            "size": n_results
        }
    )

    chunks = []
    for rank, hit in enumerate(resp["hits"]["hits"]):
        src = hit["_source"]
        chunks.append({
            "chunk_id": src["chunk_id"],
            "text": src["text"],
            "metadata": {
                "source":      src.get("source", ""),
                "file_path":   src.get("file_path", ""),
                "file_type":   src.get("file_type", ""),
                "category":    src.get("category", ""),
                "ingested_at": src.get("ingested_at", ""),
                "page":        src.get("page", 0),
                "total_pages": src.get("total_pages", 0),
                "chunk_id":    src.get("chunk_id", ""),
                "chunk_index": src.get("chunk_index", 0),
                "word_count":  src.get("word_count", 0),
            },
            "score": hit["_score"],
            "rank": rank
        })

    return chunks