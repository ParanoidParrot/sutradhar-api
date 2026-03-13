"""
ingest.py
Document ingestion pipeline for Sutradhar.
Supports: PDF (digital + scanned via OCR), plain text (.txt), Word (.docx), web URLs

Extraction strategy for PDFs:
  1. Try pymupdf (fast, handles digital PDFs + Indic Unicode)
  2. If no text extracted → scanned PDF → OCR via Tesseract (capped at OCR_PAGE_LIMIT pages)

Usage:
  python3 ingest.py --file path/to/file.pdf --scripture ramayana --source "Valmiki Ramayana Vol 1" --kanda "Bala Kanda"
  python3 ingest.py --url https://example.com/ramayana --scripture ramayana --source "Sacred Texts"
  python3 ingest.py --seed  ← seeds from data/passages.json
"""

import os
import json
import uuid
import argparse
import requests
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "sutradhar")
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))

pc = Pinecone(api_key=PINECONE_API_KEY)

CHUNK_SIZE     = 400   # words per chunk
CHUNK_OVERLAP  = 50    # words overlap between chunks
BATCH_SIZE     = 50    # vectors per Pinecone upsert batch
EMBED_BATCH    = 50    # chunks per Pinecone inference embed call
OCR_PAGE_LIMIT = 20    # max pages to OCR for scanned PDFs


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_from_pdf(path: str) -> tuple[str, str]:
    """
    Extract text from PDF.
    Returns (text, method) where method is 'digital' or 'ocr'.
    Raises warning if OCR page limit is hit.
    """
    import fitz  # pymupdf

    doc        = fitz.open(path)
    total_pages = len(doc)
    pages_text  = []

    # Step 1 — try digital extraction
    for page in doc:
        pages_text.append(page.get_text())

    extracted = "\n".join(pages_text).strip()

    if extracted:
        print(f"  PDF: digital text extracted from {total_pages} pages")
        return extracted, "digital"

    # Step 2 — no text found, try OCR
    print(f"  PDF: no digital text found — scanned PDF detected")
    print(f"  OCR: processing up to {OCR_PAGE_LIMIT} of {total_pages} pages")

    if total_pages > OCR_PAGE_LIMIT:
        print(f"  ⚠️  WARNING: PDF has {total_pages} pages but OCR is capped at {OCR_PAGE_LIMIT}.")
        print(f"  ⚠️  To ingest the full document, split it into {OCR_PAGE_LIMIT}-page chunks first.")

    try:
        import pytesseract
        from PIL import Image
        import io
        from concurrent.futures import ThreadPoolExecutor, as_completed

        pages_to_ocr = min(total_pages, OCR_PAGE_LIMIT)
        lang_str     = "eng+hin+tam+tel+kan+mal+ben+mar+guj+pan+ori"

        def ocr_page(page_num):
            # Each thread gets its own page render — fitz pages are not thread-safe
            # so we re-open the doc per thread
            _doc = fitz.open(path)
            page = _doc[page_num]
            mat  = fitz.Matrix(150 / 72, 150 / 72)
            pix  = page.get_pixmap(matrix=mat)
            img  = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang=lang_str)
            _doc.close()
            print(f"  OCR: page {page_num + 1}/{pages_to_ocr} done")
            return page_num, text.strip()

        # Run OCR on all pages in parallel — 10 threads
        ocr_results = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(ocr_page, i): i for i in range(pages_to_ocr)}
            for future in as_completed(futures):
                page_num, text = future.result()
                if text:
                    ocr_results[page_num] = text

        # Reassemble in original page order
        result = "\n".join(ocr_results[i] for i in sorted(ocr_results))
        if not result:
            raise ValueError("OCR ran but extracted no text. The PDF may contain only images or graphics.")
        return result, "ocr"

    except ImportError:
        raise ImportError("pytesseract or Pillow not installed. Run: pip install pytesseract Pillow")


def extract_from_docx(path: str) -> str:
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except ImportError:
        raise ImportError("Run: pip install python-docx")


def extract_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_from_url(url: str) -> str:
    try:
        from bs4 import BeautifulSoup
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except ImportError:
        raise ImportError("Run: pip install beautifulsoup4")


def extract_text(source: str, is_url: bool = False) -> str:
    if is_url:
        return extract_from_url(source)
    ext = os.path.splitext(source)[1].lower()
    if ext == ".pdf":
        text, method = extract_from_pdf(source)
        return text
    elif ext == ".docx":
        return extract_from_docx(source)
    elif ext == ".txt":
        return extract_from_txt(source)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .txt, .docx")


def extract_text_with_meta(source: str, is_url: bool = False) -> dict:
    """Extended version of extract_text that also returns extraction metadata."""
    if is_url:
        return {"text": extract_from_url(source), "method": "url", "ocr_capped": False}
    ext = os.path.splitext(source)[1].lower()
    if ext == ".pdf":
        import fitz
        doc         = fitz.open(source)
        total_pages = len(doc)
        text, method = extract_from_pdf(source)
        ocr_capped  = method == "ocr" and total_pages > OCR_PAGE_LIMIT
        return {
            "text":        text,
            "method":      method,
            "total_pages": total_pages,
            "ocr_capped":  ocr_capped,
            "ocr_limit":   OCR_PAGE_LIMIT,
        }
    elif ext == ".docx":
        return {"text": extract_from_docx(source), "method": "docx", "ocr_capped": False}
    elif ext == ".txt":
        return {"text": extract_from_txt(source), "method": "txt", "ocr_capped": False}
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


# ── Pinecone embedding — batched to avoid 413 ────────────────────────────────
def embed_chunks(chunks: list[str]) -> list[list[float]]:
    all_vectors = []
    total_batches = -(-len(chunks) // EMBED_BATCH)  # ceiling division
    for i in range(0, len(chunks), EMBED_BATCH):
        batch  = chunks[i:i + EMBED_BATCH]
        result = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=batch,
            parameters={"input_type": "passage"}
        )
        all_vectors.extend([r.values for r in result])
        print(f"  Embedded batch {i // EMBED_BATCH + 1}/{total_batches} ({len(batch)} chunks)")
    return all_vectors


# ── Pinecone upsert — batched ─────────────────────────────────────────────────
def upsert_to_pinecone(
    chunks:    list[str],
    namespace: str,
    source:    str,
    kanda:     str = "",
    topic:     str = ""
) -> int:
    index   = pc.Index(PINECONE_INDEX)
    vectors = embed_chunks(chunks)

    records = []
    for chunk, vector in zip(chunks, vectors):
        records.append({
            "id":     str(uuid.uuid4()),
            "values": vector,
            "metadata": {
                "text":   chunk,
                "source": source,
                "kanda":  kanda,
                "topic":  topic
            }
        })

    total_batches = -(-len(records) // BATCH_SIZE)
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"  Upserted batch {i // BATCH_SIZE + 1}/{total_batches} ({len(batch)} vectors)")

    return len(records)


# ── Seed from passages.json ───────────────────────────────────────────────────
def seed_from_json():
    passages_file = os.path.join(BASE_DIR, "data", "passages.json")
    if not os.path.exists(passages_file):
        print(f"ERROR: {passages_file} not found.")
        return

    with open(passages_file, "r", encoding="utf-8") as f:
        passages = json.load(f)

    print(f"Seeding {len(passages)} passages from passages.json...")

    index   = pc.Index(PINECONE_INDEX)
    vectors = embed_chunks([p["text"] for p in passages])

    records = []
    for passage, vector in zip(passages, vectors):
        records.append({
            "id":     passage["id"],
            "values": vector,
            "metadata": {
                "text":   passage["text"],
                "source": "Valmiki Ramayana",
                "kanda":  passage["kanda"],
                "topic":  passage["topic"]
            }
        })

    total_batches = -(-len(records) // BATCH_SIZE)
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace="ramayana")
        print(f"  Upserted batch {i // BATCH_SIZE + 1}/{total_batches} ({len(batch)} vectors)")

    print(f"\n✅ Seeded {len(records)} passages into Pinecone namespace: ramayana")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sutradhar document ingestion")
    parser.add_argument("--file",      help="Path to PDF, TXT or DOCX file")
    parser.add_argument("--url",       help="Web URL to ingest")
    parser.add_argument("--scripture", default="ramayana")
    parser.add_argument("--source",    default="")
    parser.add_argument("--kanda",     default="")
    parser.add_argument("--topic",     default="")
    parser.add_argument("--seed",      action="store_true")
    args = parser.parse_args()

    with open(os.path.join(BASE_DIR, "persistent", "scriptures.json")) as f:
        scriptures = {s["id"]: s for s in json.load(f)["scriptures"]}

    if args.seed:
        seed_from_json()
        return

    if not args.file and not args.url:
        print("ERROR: Provide --file, --url, or --seed")
        parser.print_help()
        return

    if args.scripture not in scriptures:
        print(f"ERROR: Unknown scripture '{args.scripture}'. Available: {list(scriptures.keys())}")
        return

    namespace = scriptures[args.scripture]["pinecone_namespace"]
    is_url    = bool(args.url)
    source    = args.url or args.file

    print(f"Extracting text from: {source}")
    meta   = extract_text_with_meta(source, is_url=is_url)
    text   = meta["text"]

    if meta.get("ocr_capped"):
        print(f"\n⚠️  OCR capped at {OCR_PAGE_LIMIT} pages (PDF has {meta['total_pages']} pages total)")
        print(f"⚠️  Only the first {OCR_PAGE_LIMIT} pages were ingested\n")

    print(f"Extracted {len(text.split())} words via {meta['method']}")
    print(f"Chunking into ~{CHUNK_SIZE} word chunks...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")

    print(f"Embedding and upserting to Pinecone namespace: {namespace}")
    total = upsert_to_pinecone(
        chunks=chunks,
        namespace=namespace,
        source=args.source or source,
        kanda=args.kanda,
        topic=args.topic
    )

    print(f"\n✅ Successfully ingested {total} chunks into Pinecone!")
    print(f"   Scripture : {args.scripture}")
    print(f"   Namespace : {namespace}")
    print(f"   Source    : {args.source or source}")
    if meta.get("method") == "ocr":
        print(f"   Method    : OCR (scanned PDF)")


if __name__ == "__main__":
    main()