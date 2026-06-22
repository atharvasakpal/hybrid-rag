SYSTEM_PROMPT = """You are an enterprise knowledge assistant.
Answer only from the provided context.
Use citations in square brackets like [S1] or [S2] for every factual claim.
If the context does not contain enough evidence, say that the provided documents do not contain enough information.
Do not invent sources, page numbers, or citations.
Never use numeric paper references like [3], [15], or [29] as citations.
Only use source labels in the format [S1], [S2], [S3]."""


def build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Convert retrieved chunks into labeled context blocks and citation metadata.
    """
    context_blocks = []
    citations = []

    for index, chunk in enumerate(chunks, start=1):
        label = f"S{index}"
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")

        citations.append({
            "label": label,
            "source": source,
            "page": page,
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text", ""),
            "rerank_score": chunk.get("rerank_score"),
            "rrf_score": chunk.get("rrf_score"),
        })

        context_blocks.append(
            f"[{label}] Source: {source} | Page: {page}\n"
            f"{chunk.get('text', '')}"
        )

    return "\n\n---\n\n".join(context_blocks), citations


def build_messages(query: str, chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    context, citations = build_context(chunks)
    user_prompt = f"""Question:
{query}

Context:
{context}

Instructions:
- Answer in a concise, professional style.
- Cite every factual claim with source labels from the context, such as [S1].
- Use separate citation labels like [S1] [S2], not combined labels like [S1, S2].
- Do not use numeric references copied from the paper text, such as [3], [15], or [29].
- Use only source labels that appear in the context.
- If the answer is uncertain or incomplete from the context, state the limitation clearly."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    return messages, citations
