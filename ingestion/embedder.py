import os
from dotenv import load_dotenv
import chromadb
from tqdm import tqdm

load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = "medical_rag"
BATCH_SIZE = 50


from functools import lru_cache

@lru_cache(maxsize=1)
def get_chroma_client():
    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
    )

def get_or_create_collection(client):
    """Get existing collection or create new one."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for MiniLM
    )


def embed_and_store(chunks: list[dict], embeddings_model):
    """
    Embed chunks and store in ChromaDB.
    Skips chunks that already exist by chunk_id.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # find which chunks are already stored
    existing = set()
    try:
        existing_data = collection.get(include=[])
        existing = set(existing_data["ids"])
        print(f"Found {len(existing)} existing chunks in ChromaDB")
    except Exception:
        pass

    # filter to only new chunks
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing]
    print(f"New chunks to embed and store: {len(new_chunks)}")

    if not new_chunks:
        print("Nothing to add — all chunks already in ChromaDB")
        return

    # process in batches
    for i in tqdm(range(0, len(new_chunks), BATCH_SIZE), desc="Embedding batches"):
        batch = new_chunks[i: i + BATCH_SIZE]

        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        # generate embeddings for this batch
        embeddings = embeddings_model.embed_documents(texts)

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    print(f"✓ ChromaDB now contains {collection.count()} chunks")
