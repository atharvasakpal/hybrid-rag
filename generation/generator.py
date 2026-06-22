import os
import re
import time
from functools import lru_cache

from dotenv import load_dotenv
# from groq import Groq, APITimeoutError, APIConnectionError   # COMMENTED OUT
from google import genai
from google.genai import types
from google.api_core.exceptions import DeadlineExceeded, ServiceUnavailable

from generation.prompt import build_messages
from reranking.reranker import retrieve_and_rerank

load_dotenv()

# GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")   # COMMENTED OUT
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 700
MAX_CITATION_RETRIES = 1
GROQ_TIMEOUT_SECONDS = 10.0   # reused as GEMINI_TIMEOUT_SECONDS below
MAX_TIMEOUT_RETRIES = 1


# @lru_cache(maxsize=1)                                            # COMMENTED OUT
# def get_groq_client():
#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         raise ValueError("GROQ_API_KEY is not set")
#     return Groq(api_key=api_key, timeout=GROQ_TIMEOUT_SECONDS)


@lru_cache(maxsize=1)
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def validate_citations(answer: str, citations: list[dict]) -> dict:
    # unchanged — no edits needed here
    allowed_labels = {citation["label"] for citation in citations}
    bracketed_labels = set(re.findall(r"\[([^\[\]]+)\]", answer))
    used_labels = set()
    invalid_labels = set()

    for bracketed_label in bracketed_labels:
        labels = [
            label.strip()
            for label in bracketed_label.split(",")
            if label.strip()
        ]

        if labels and all(re.fullmatch(r"S\d+", label) for label in labels):
            unknown_labels = set(labels) - allowed_labels
            if unknown_labels:
                invalid_labels.update(unknown_labels)
            else:
                used_labels.update(labels)
        else:
            invalid_labels.add(bracketed_label)

    return {
        "valid": not invalid_labels and bool(used_labels),
        "used_labels": sorted(used_labels),
        "invalid_labels": sorted(invalid_labels),
        "available_labels": sorted(allowed_labels),
    }


# def _call_groq(                                                  # COMMENTED OUT
#     messages: list[dict],
#     model: str,
#     temperature: float,
#     max_tokens: int
# ) -> str:
#     client = get_groq_client()
#     for attempt in range(MAX_TIMEOUT_RETRIES + 1):
#         try:
#             response = client.chat.completions.create(
#                 model=model,
#                 messages=messages,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#             )
#             return response.choices[0].message.content
#         except (APITimeoutError, APIConnectionError):
#             if attempt < MAX_TIMEOUT_RETRIES:
#                 print(f"  ⚠ Groq call timed out/failed (attempt {attempt + 1}), retrying...")
#                 continue
#             raise


def _call_gemini(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int
) -> str:
    """
    Gemini replacement for _call_groq. Converts OpenAI-style messages
    (system/user/assistant) into Gemini's expected format: a separate
    system_instruction plus a content history of user/model turns.
    """
    client = get_gemini_client()

    system_instruction = ""
    contents = []
    for m in messages:
        if m["role"] == "system":
            system_instruction = m["content"]
        elif m["role"] == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
        elif m["role"] == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=m["content"])]))

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    for attempt in range(MAX_TIMEOUT_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response.text
        except (DeadlineExceeded, ServiceUnavailable):
            if attempt < MAX_TIMEOUT_RETRIES:
                print(f"  ⚠ Gemini call timed out/failed (attempt {attempt + 1}), retrying...")
                continue
            raise


def generate_answer(
    query: str,
    chunks: list[dict],
    model: str = GEMINI_MODEL,                                    # CHANGED from GROQ_MODEL
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> dict:
    """
    Generate a grounded answer from already-reranked chunks.
    """
    if not chunks:
        return {
            "query": query,
            "answer": "The provided documents do not contain enough information to answer this question.",
            "citations": [],
            "citation_validation": {
                "valid": False,
                "used_labels": [],
                "invalid_labels": [],
                "available_labels": [],
            },
            "model": model,
            "latency_seconds": 0.0,
        }

    messages, citations = build_messages(query, chunks)
    start_time = time.time()
    answer = _call_gemini(messages, model, temperature, max_tokens)   # CHANGED from _call_groq
    citation_validation = validate_citations(answer, citations)

    for _ in range(MAX_CITATION_RETRIES):
        if citation_validation["valid"]:
            break

        messages.append({
            "role": "assistant",
            "content": answer,
        })
        messages.append({
            "role": "user",
            "content": (
                "Rewrite the answer using only the available citation labels "
                f"{citation_validation['available_labels']}. "
                "Do not use numeric paper references like [3] or [15]. "
                "Every factual claim must cite labels like [S1]."
            ),
        })
        answer = _call_gemini(messages, model, temperature, max_tokens)  # CHANGED from _call_groq
        citation_validation = validate_citations(answer, citations)

    latency_seconds = time.time() - start_time

    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "citation_validation": citation_validation,
        "model": model,
        "latency_seconds": latency_seconds,
    }


def answer_query(
    query: str,
    hybrid_k: int = 20,
    rerank_k: int = 5,
    rerank_provider: str = "cohere",
    model: str = GEMINI_MODEL                                     # CHANGED from GROQ_MODEL
) -> dict:
    """
    Full RAG path:
    hybrid retrieval -> reranking -> grounded generation with citations.
    """
    start_time = time.time()
    chunks = retrieve_and_rerank(
        query=query,
        hybrid_k=hybrid_k,
        top_k=rerank_k,
        provider=rerank_provider
    )

    result = generate_answer(
        query=query,
        chunks=chunks,
        model=model
    )
    result["retrieved_chunks"] = chunks
    result["total_latency_seconds"] = time.time() - start_time

    return result