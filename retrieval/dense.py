import os
from functools import lru_cache

from dotenv import load_dotenv
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = "medical_rag"


def get_chroma_client():
    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
    )


@lru_cache(maxsize=1)
def get_embeddings_model(model_name: str = "all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def query_chroma(query: str, n_results: int = 20) -> list[dict]:
    """
    Dense retrieval from ChromaDB.
    Returns top n_results chunks with text, metadata, score, rank.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    embeddings_model = get_embeddings_model()

    query_embedding = embeddings_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i],
            "rank": i
        })

    return chunks
