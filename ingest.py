"""
ingest.py
Document ingestion pipeline for Sutradhar.
Supports: PDF, plain text (.txt), Word (.docx), web URLs
Chunks documents, embeds via Pinecone inference, upserts to Pinecone.

Usage:
  python3 ingest.py --file path/to/file.pdf --scripture ramayana --source "Valmiki Ramayana Vol 1" --kanda "Bala Kanda"
  python3 ingest.py --url https://example.com/ramayana --scripture ramayana --source "Sacred Texts"
  python3 ingest.py --seed  ← seeds from data/passages.json (same as prototype)
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

CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 50    # words overlap between chunks
BATCH_SIZE    = 90    # Pinecone upsert batch size


# ── Text extraction ───────────────────────────────────────────────────────────
def extract_from_pdf(path: str) -> str:
    try:
        import PyPDF2
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    except ImportError:
        raise ImportError("Run: pip install pypdf2")


def extract_from_docx(path: str) -> str:
    try:
        from docx import Document
        doc   = Document(path)
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
        # Remove script and style tags
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
        return extract_from_pdf(source)
    elif ext == ".docx":
        return extract_from_docx(source)
    elif ext == ".txt":
        return extract_from_txt(source)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .txt, .docx")


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


# ── Pinecone embedding + upsert ───────────────────────────────────────────────
def embed_chunks(chunks: list[str]) -> list[list[float]]:
    result = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=chunks,
        parameters={"input_type": "passage"}
    )
    return [r.values for r in result]


def upsert_to_pinecone(
    chunks:    list[str],
    namespace: str,
    source:    str,
    kanda:     str = "",
    topic:     str = ""
):
    index   = pc.Index(PINECONE_INDEX)
    vectors = embed_chunks(chunks)

    records = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
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

    # Upsert in batches
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"  Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} vectors)")

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

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace="ramayana")
        print(f"  Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} vectors)")

    print(f"\n✅ Seeded {len(records)} passages into Pinecone namespace: ramayana")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sutradhar document ingestion")
    parser.add_argument("--file",       help="Path to PDF, TXT or DOCX file")
    parser.add_argument("--url",        help="Web URL to ingest")
    parser.add_argument("--scripture",  default="ramayana", help="Scripture namespace (default: ramayana)")
    parser.add_argument("--source",     default="", help="Source name e.g. 'Valmiki Ramayana Vol 1'")
    parser.add_argument("--kanda",      default="", help="Kanda/section name")
    parser.add_argument("--topic",      default="", help="Topic tag")
    parser.add_argument("--seed",       action="store_true", help="Seed from data/passages.json")
    args = parser.parse_args()

    with open(os.path.join(BASE_DIR, "scriptures.json")) as f:
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
    text   = extract_text(source, is_url=is_url)
    print(f"Extracted {len(text.split())} words")

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


if __name__ == "__main__":
    main()