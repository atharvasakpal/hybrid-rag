import time
from pathlib import Path

import requests
from lxml import etree
from datetime import datetime

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BATCH_SIZE = 100
SLEEP_BETWEEN_BATCHES_SECONDS = 1.0  # E-Utilities allows ~3 req/s without an API key
OUTPUT_DIR = Path("data/pmc_papers")


def search_pmc_oa(query: str, max_results: int = 2000) -> list[str]:
    params = {
        "db": "pmc",
        "term": f"{query} AND open access[filter]",
        "retmax": max_results,
        "retmode": "json",
    }
    resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]


def fetch_pmc_xml(ids: list[str]) -> bytes:
    params = {
        "db": "pmc",
        "id": ",".join(ids),
        "retmode": "xml",
    }
    resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=60)
    resp.raise_for_status()
    return resp.content


def parse_pmc_articleset(xml_bytes: bytes, category: str = "medical") -> list[dict]:
    root = etree.fromstring(xml_bytes)
    articles = root.findall(".//article")
    results = []

    for article in articles:
        pmcid_el = article.find(".//article-id[@pub-id-type='pmcid']")
        pmcid = pmcid_el.text if pmcid_el is not None else "unknown"

        title_el = article.find(".//article-title")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        abstract_paragraphs = []
        for p in article.findall(".//abstract//p"):
            text = "".join(p.itertext()).strip()
            if text:
                abstract_paragraphs.append(text)

        body_paragraphs = []
        for p in article.findall(".//body//p"):
            text = "".join(p.itertext()).strip()
            if text:
                body_paragraphs.append(text)

        full_text = title + "\n\n" + "\n\n".join(abstract_paragraphs + body_paragraphs)

        if len(full_text.strip()) < 100:
            continue

        results.append({
            "pmcid": pmcid,
            "text": full_text.strip(),
        })

    return results


def fetch_and_save_corpus(query: str, max_results: int = 2000):
    """
    Search PMC OA, fetch full text in batches, save each article as a
    plain .txt file. Designed to be resumable/fault-tolerant — a failed
    batch is logged and skipped, not fatal to the whole run.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Searching PMC OA for: {query}")
    ids = search_pmc_oa(query, max_results=max_results)
    print(f"Found {len(ids)} article IDs")

    batches = [ids[i:i + BATCH_SIZE] for i in range(0, len(ids), BATCH_SIZE)]
    print(f"Processing in {len(batches)} batches of up to {BATCH_SIZE}")

    saved_count = 0
    failed_batches = []
    start_time = time.time()

    for batch_index, batch_ids in enumerate(batches, start=1):
        try:
            xml_bytes = fetch_pmc_xml(batch_ids)
            articles = parse_pmc_articleset(xml_bytes)

            for article in articles:
                out_path = OUTPUT_DIR / f"{article['pmcid']}.txt"
                out_path.write_text(article["text"], encoding="utf-8")
                saved_count += 1

            print(f"[{batch_index}/{len(batches)}] Saved {len(articles)}/{len(batch_ids)} articles from this batch")

        except Exception as e:
            print(f"[{batch_index}/{len(batches)}] Batch failed: {e}")
            failed_batches.append(batch_index)

        time.sleep(SLEEP_BETWEEN_BATCHES_SECONDS)

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"DONE. Saved {saved_count} articles in {elapsed:.1f}s")
    print(f"Failed batches: {failed_batches if failed_batches else 'none'}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    fetch_and_save_corpus("diabetes", max_results=2000)