import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import pypdf
from docx import Document


def load_document(file_path: str, category: str = "medical") -> list[dict]:
    """
    Load a document and return list of pages with metadata.
    Each item: {"text": str, "metadata": dict}
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = path.suffix.lower()
    
    if ext == ".pdf":
        return _load_pdf(path, category)
    elif ext == ".docx":
        return _load_docx(path, category)
    elif ext == ".txt":
        return _load_txt(path, category)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _base_metadata(path: Path, file_type: str, category: str) -> dict:
    """Metadata common to all file types."""
    return {
        "source": path.name,
        "file_path": str(path),
        "file_type": file_type,
        "category": category,
        "ingested_at": datetime.utcnow().isoformat(),
    }


def _load_pdf(path: Path, category: str) -> list[dict]:
    pages = []
    
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        total_pages = len(reader.pages)
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            
            # skip empty pages
            if not text or len(text.strip()) < 50:
                continue
            
            metadata = _base_metadata(path, "pdf", category)
            metadata["page"] = page_num + 1
            metadata["total_pages"] = total_pages
            
            pages.append({
                "text": text.strip(),
                "metadata": metadata
            })
    
    return pages


def _load_docx(path: Path, category: str) -> list[dict]:
    doc = Document(str(path))
    
    # docx has no pages — treat whole document as one unit
    full_text = "\n".join([
        para.text for para in doc.paragraphs 
        if para.text.strip()
    ])
    
    if not full_text.strip():
        return []
    
    metadata = _base_metadata(path, "docx", category)
    metadata["page"] = 1
    metadata["total_pages"] = 1
    
    return [{"text": full_text.strip(), "metadata": metadata}]


def _load_txt(path: Path, category: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    if not text.strip():
        return []
    
    metadata = _base_metadata(path, "txt", category)
    metadata["page"] = 1
    metadata["total_pages"] = 1
    
    return [{"text": text.strip(), "metadata": metadata}]


def load_all_documents(folder_path: str, category: str = "medical") -> list[dict]:
    """Load all supported documents from a folder."""
    folder = Path(folder_path)
    supported = {".pdf", ".docx", ".txt"}
    all_pages = []
    
    files = [f for f in folder.iterdir() if f.suffix.lower() in supported]
    
    print(f"Found {len(files)} documents in {folder_path}")
    
    for file in files:
        try:
            pages = load_document(str(file), category)
            all_pages.extend(pages)
            print(f"  ✓ {file.name} — {len(pages)} pages loaded")
        except Exception as e:
            print(f"  ✗ {file.name} — failed: {e}")
    
    print(f"\nTotal pages loaded: {len(all_pages)}")
    return all_pages