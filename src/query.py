#!/usr/bin/env python3
"""Query the Qdrant vector database for relevant transcript chunks.

This script connects to an existing Qdrant collection and performs
semantic similarity search based on a user query.

Usage:
    # With command-line argument
    uv run python src/query.py "What does Michael Harris think about leadership?"

    # With environment variable
    QUERY="Laura Bennett's approach to change" uv run python src/query.py

    # With options
    uv run python src/query.py "Daniel Wright" --top-k 5

Environment Variables:
    OPENAI_API_KEY: Required. Your OpenAI API key.
    QDRANT_URL: Qdrant server URL (default: http://localhost:6333)
    QDRANT_COLLECTION: Collection name (default: mentions_mvp)
    QUERY: Default query if not provided as argument
    TOP_K: Number of results to return (default: 3)
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# Configuration from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "mentions_mvp")

# Default query uses real names from sample transcript
DEFAULT_QUERY = "What is Michael Harris's approach to leadership?"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Query Qdrant for relevant transcript chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Michael Harris leadership style"
  %(prog)s "Laura Bennett's decisions" --top-k 5
  %(prog)s "Daniel Wright as advisor" --verbose
        """,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=os.environ.get("QUERY", DEFAULT_QUERY),
        help="Search query (default: from QUERY env var or built-in default)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=int(os.environ.get("TOP_K", "3")),
        help="Number of results to return (default: 3)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show full chunk text (default: truncated to 200 chars)",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION,
        help=f"Qdrant collection name (default: {COLLECTION})",
    )
    return parser.parse_args()


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max_length, adding ellipsis if truncated."""
    text = text.strip().replace("\n", " ")
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def main() -> int:
    """Main entry point for the query script."""
    args = parse_args()

    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        print("Please set it in your environment or .env file.", file=sys.stderr)
        return 1

    # Connect to embeddings and Qdrant
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    try:
        store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=QDRANT_URL,
            collection_name=args.collection,
        )
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}", file=sys.stderr)
        print(f"Make sure Qdrant is running at {QDRANT_URL}", file=sys.stderr)
        print(f"and collection '{args.collection}' exists.", file=sys.stderr)
        return 1

    # Perform similarity search
    results = store.similarity_search_with_score(args.query, k=args.top_k)

    # Display results
    print(f"\nQuery: {args.query}")
    print(f"Collection: {args.collection}")
    print(f"Results: {len(results)}")
    print("-" * 60)

    if not results:
        print("No results found.")
        return 0

    for rank, (doc, score) in enumerate(results, start=1):
        md = doc.metadata
        print(f"\n#{rank}  score: {score:.4f}")
        print(f"    source: {md.get('source_file', 'unknown')}")
        print(f"    chunk_id: {md.get('chunk_id', 'N/A')}")
        print(f"    video_url: {md.get('video_url', 'N/A')}")

        if md.get("start_sec") is not None:
            print(f"    time: {md.get('start_sec')}s - {md.get('end_sec')}s")

        # Display text content
        text = doc.page_content
        if args.verbose:
            # Show full text with indentation
            print("    text:")
            for line in text.strip().split("\n"):
                print(f"      {line}")
        else:
            # Show truncated text
            print(f"    text: {truncate_text(text)}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
