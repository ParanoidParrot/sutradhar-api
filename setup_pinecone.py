"""
setup_pinecone.py
Creates the Pinecone index for Sutradhar. Run once before ingesting documents.
Usage: python3 setup_pinecone.py
"""

import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "sutradhar")

def setup():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX in existing:
        print(f"Index '{PINECONE_INDEX}' already exists — nothing to do.")
        return

    print(f"Creating Pinecone index: {PINECONE_INDEX}")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1024,         # multilingual-e5-large dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # free tier region
        )
    )
    print(f"✅ Index '{PINECONE_INDEX}' created successfully.")
    print(f"\nNext steps:")
    print(f"  1. Run: python3 ingest.py --seed")
    print(f"  2. Run: uvicorn main:app --reload")

if __name__ == "__main__":
    setup()