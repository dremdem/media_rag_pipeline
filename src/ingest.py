#!/usr/bin/env python3
"""Ingest transcript files into Qdrant vector database.

This script reads a transcript file, splits it into chunks, generates
embeddings using OpenAI, and stores them in a Qdrant collection.

Usage:
    uv run python src/ingest.py

Environment Variables:
    OPENAI_API_KEY: Required. Your OpenAI API key.
    QDRANT_URL: Qdrant server URL (default: http://localhost:6333)
    QDRANT_COLLECTION: Collection name (default: mentions_mvp)
    TRANSCRIPT_PATH: Path to transcript file (default: data/transcripts/sample.txt)
    VIDEO_URL: Optional video URL for metadata
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Configuration from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "mentions_mvp")
TRANSCRIPT_PATH = os.environ.get("TRANSCRIPT_PATH", "data/transcripts/sample.txt")


def main() -> None:
    """Main entry point for the ingest script."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Please set it in your environment or .env file."
        )

    path = Path(TRANSCRIPT_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")

    print(f"Reading transcript: {path}")
    raw_text = path.read_text(encoding="utf-8")
    print(f"  Characters: {len(raw_text)}")

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    docs = splitter.create_documents([raw_text])
    print(f"  Chunks created: {len(docs)}")

    # Attach metadata to each chunk
    video_url = os.environ.get("VIDEO_URL", "https://youtube.com/watch?v=VIDEO_ID")
    for i, doc in enumerate(docs):
        doc.metadata.update(
            {
                "source_file": str(path),
                "chunk_id": i,
                "video_url": video_url,
                # Placeholders for timestamps (to be implemented in later phases)
                "start_sec": None,
                "end_sec": None,
            }
        )

    # Create embeddings and store in Qdrant
    print(f"Generating embeddings with OpenAI...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print(f"Storing in Qdrant collection: {COLLECTION}")
    print(f"  Qdrant URL: {QDRANT_URL}")

    QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION,
    )

    print(f"Ingested {len(docs)} chunks into Qdrant collection: {COLLECTION}")


if __name__ == "__main__":
    main()
