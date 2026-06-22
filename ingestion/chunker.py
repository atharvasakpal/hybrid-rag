import hashlib
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    """Load the embedding model once and reuse."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def generate_chunk_id(source: str, page: int, chunk_index: int) -> str:
    """
    Generate a unique, deterministic ID for each chunk.
    Same chunk always gets same ID — important for RRF deduplication.
    Format: hash of source+page+index so it's short but unique.
    """
    raw = f"{source}_{page}_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def chunk_pages(
    pages: list[dict],
    embeddings_model,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 85.0
) -> list[dict]:
    """
    Takes pages from loader, returns semantic chunks with metadata.
    
    breakpoint_threshold_amount: higher = fewer, larger chunks
                                 lower = more, smaller chunks
    85 is a good starting point for medical text.
    """
    splitter = SemanticChunker(
        embeddings_model,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount
    )
    
    all_chunks = []
    
    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        
        # semantic chunker returns LangChain Document objects
        docs = splitter.create_documents([text])
        
        for chunk_index, doc in enumerate(docs):
            chunk_text = doc.page_content.strip()
            
            # skip very short chunks — usually headers or noise
            if len(chunk_text.split()) < 20:
                continue
            
            chunk_id = generate_chunk_id(
                metadata["source"],
                metadata["page"],
                chunk_index
            )
            
            # inherit all page metadata + add chunk-specific fields
            chunk_metadata = {
                **metadata,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "word_count": len(chunk_text.split()),
            }
            
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": chunk_metadata
            })
    
    return all_chunks
