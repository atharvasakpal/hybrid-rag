import os
import time
from pathlib import Path
from dotenv import load_dotenv

from ingestion.loader import load_document
from ingestion.chunker import chunk_pages, get_embeddings
from ingestion.embedder import embed_and_store
from retrieval.sparse import index_chunks, create_index_if_not_exists, get_es_client

load_dotenv()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def run_pipeline(
    folder_path: str,
    category: str = "medical",
    model_name: str = "all-MiniLM-L6-v2"
):
    """
    Full ingestion pipeline: load → chunk → embed → index.
    Processes one document at a time for fault tolerance.
    """
    folder = Path(folder_path)
    files = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        print(f"No supported documents found in {folder_path}")
        return

    print(f"Found {len(files)} documents to process")
    print("=" * 50)

    # load embedding model once — reused across all documents
    print("Loading embedding model...")
    embeddings_model = get_embeddings(model_name)

    # ensure elasticsearch index exists before processing
    es = get_es_client()
    create_index_if_not_exists(es)

    results = {"success": [], "failed": []}
    total_chunks = 0
    start_time = time.time()

    for i, file in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {file.name}")
        file_start = time.time()

        try:
            # step 1: load
            pages = load_document(str(file), category)
            if not pages:
                print(f"  ⚠ No extractable text — skipping")
                results["failed"].append((file.name, "no extractable text"))
                continue

            # step 2: chunk
            chunks = chunk_pages(pages, embeddings_model)
            if not chunks:
                print(f"  ⚠ No chunks produced — skipping")
                results["failed"].append((file.name, "no chunks produced"))
                continue

            # step 3: embed and store in ChromaDB
            embed_and_store(chunks, embeddings_model)

            # step 4: index in Elasticsearch
            index_chunks(chunks)

            elapsed = time.time() - file_start
            total_chunks += len(chunks)
            results["success"].append(file.name)
            print(f"  ✓ {len(chunks)} chunks | {elapsed:.1f}s")

        except Exception as e:
            results["failed"].append((file.name, str(e)))
            print(f"  ✗ Failed: {e}")
            continue

    # summary
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"PIPELINE COMPLETE")
    print(f"  Processed:  {len(results['success'])}/{len(files)} documents")
    print(f"  Total chunks created: {total_chunks}")
    print(f"  Total time: {total_time:.1f}s")

    if results["failed"]:
        print(f"\n  Failed documents:")
        for name, reason in results["failed"]:
            print(f"    ✗ {name}: {reason}")

    return results


if __name__ == "__main__":
    run_pipeline("data/papers/")
