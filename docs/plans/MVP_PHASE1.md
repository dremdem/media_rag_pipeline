# MVP Phase 1: Local RAG Prototype (Qdrant OSS + OpenAI Embeddings + LangChain)

Goal: build a **very rough MVP** on a local dev machine (MacBook Air M1, 8GB RAM) that can:
1) **Ingest** a transcript text file → create embeddings → store in **Qdrant OSS** (local Docker)  
2) **Query** with a text prompt → retrieve relevant chunks → print results to console

We are **not** automating YouTube/transcription yet — transcripts and video metadata are added manually.

---

## Architecture (Phase 1)

- **Vector DB:** Qdrant OSS (self-hosted locally in Docker)
- **Embeddings:** OpenAI Embeddings API
- **Orchestration / Integration:** LangChain (Python)
- **Input:** `./data/transcripts/<some_transcript>.txt` (plain text)
- **Output:** console logs with matched chunks + metadata (video URL, timestamps placeholders)

---

## Prerequisites

- Docker Desktop installed and running
- Python 3.11+ (recommended)
- `uv` installed (Python packaging + venv)
  - Install uv (macOS, official method):
    - If you already use uv — skip

---

## Step 1 — Run Qdrant locally (Docker)

Create a folder for persistent storage:

```bash
mkdir -p ./qdrant_storage
```

Run Qdrant:

```
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest
```

Quick health check:
```
curl -s http://localhost:6333/ | head
```

(Optional) Qdrant Web UI is usually available at:

http://localhost:6333/dashboard

Stop/remove later if needed:
```
docker stop qdrant
docker rm qdrant
```

## Step 2 — Get an OpenAI API key

Create a new folder:
```
mkdir -p rag_mvp && cd rag_mvp
```

Init project + create venv:
```
uv init
uv venv
```

uv pip install \
  langchain langchain-openai langchain-qdrant qdrant-client \
  python-dotenv tiktoken

```
rag_mvp/
  data/
    transcripts/
      sample.txt
  ingest.py
  query.py
  .env   (optional)
```

(Optional) If you prefer .env:

```
cat > .env <<'EOF'
OPENAI_API_KEY=YOUR_KEY_HERE
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=mentions_mvp
EOF
```

## Step 4 — Prepare a transcript file

### Transcript – Long-form Discussion (Approx. 10 minutes)

Persons:
- Michael Harris
- Laura Bennett
- Daniel Wright

```
mkdir -p data/transcripts
cat > data/transcripts/sample.txt <<'EOF'

Today’s discussion covers several topics related to recent developments in technology leadership, organizational change, and strategic decision-making. Throughout the conversation, we will reference insights and actions associated with Michael Harris, Laura Bennett, and Daniel Wright.

The conversation begins with a general overview of leadership transitions in mid-sized technology companies. Michael Harris is mentioned early as an example of a leader who has navigated multiple organizational changes over the last decade. According to several industry observers, Michael Harris tends to focus on incremental improvements rather than large-scale restructuring.

Later in the discussion, Laura Bennett is introduced as a contrasting example. Laura Bennett is often described as a leader who prefers decisive structural changes when she believes an organization is stagnating. Some analysts argue that Laura Bennett’s approach works well in rapidly changing markets, while others believe it introduces unnecessary risk.

At this point, Daniel Wright is mentioned in the context of advisory roles. Daniel Wright has frequently acted as a consultant during periods of leadership transition. Unlike Michael Harris, who usually stays within a single organization for long periods, Daniel Wright is known for moving between projects and companies.

The discussion then shifts to a concrete case study from two years ago. In that case, Michael Harris was responsible for overseeing a gradual modernization of internal systems. The changes were not immediately visible to external stakeholders, but internal performance metrics improved steadily. Several participants in the discussion note that this approach reflected Michael Harris’s long-term mindset.

By contrast, Laura Bennett’s involvement in a similar situation led to rapid reorganization. Within six months, reporting structures were changed, teams were merged, and some roles were eliminated. The transcript notes that Laura Bennett defended these decisions publicly, arguing that speed was essential under the circumstances.

Daniel Wright appears again at this stage as an external advisor. According to the transcript, Daniel Wright recommended a hybrid approach that combined elements of both strategies. However, his suggestions were only partially implemented, leading to mixed results.

As the conversation continues, the speaker reflects on how Michael Harris tends to communicate during periods of uncertainty. Michael Harris is described as calm and data-driven, often emphasizing measurable outcomes rather than abstract vision. This communication style is said to appeal to technical teams but may be less effective with non-technical stakeholders.

Laura Bennett’s communication style is described differently. Laura Bennett often frames decisions in terms of long-term vision and organizational identity. While this inspires some employees, others find it difficult to translate into day-to-day operational changes.

Daniel Wright is mentioned again in relation to conflict mediation. The transcript suggests that Daniel Wright frequently steps in when disagreements arise between leadership styles similar to those of Michael Harris and Laura Bennett. His role is often to clarify assumptions and identify common ground.

Midway through the discussion, the topic shifts toward decision-making under incomplete information. Michael Harris is cited as someone who prefers to delay decisions until sufficient data is available. This tendency has both supporters and critics. Some argue it reduces risk, while others believe it can slow momentum.

In contrast, Laura Bennett is again positioned as more comfortable making decisions with limited data. The transcript notes that Laura Bennett believes adaptability is more important than precision in fast-moving environments. This philosophy has shaped several of her past initiatives.

Daniel Wright’s perspective on this issue is described as pragmatic. Daniel Wright often advises leaders to explicitly define which uncertainties are acceptable and which are not. His approach is presented as a balancing mechanism between caution and speed.

Later in the transcript, the conversation returns to organizational culture. Michael Harris is associated with fostering stable, process-oriented cultures. Teams under Michael Harris are described as predictable and methodical, which can be beneficial for long-term projects.

Laura Bennett, on the other hand, is associated with cultures that emphasize experimentation and rapid feedback. The transcript mentions that under Laura Bennett’s leadership, teams are encouraged to challenge assumptions more openly, even at the cost of short-term instability.

Daniel Wright is mentioned once more as someone who evaluates cultural fit during transitions. Daniel Wright often assesses whether an organization’s existing culture aligns more closely with Michael Harris’s style or Laura Bennett’s style before making recommendations.

In the final part of the transcript, the speaker summarizes the key differences discussed. Michael Harris is portrayed as a stabilizing force, Laura Bennett as a catalyst for change, and Daniel Wright as a mediator and advisor. The transcript concludes by noting that no single approach is universally correct, and that context plays a decisive role in determining which leadership style is most effective.

The discussion ends with a brief reflection on how organizations might combine elements from all three perspectives. Michael Harris’s emphasis on data, Laura Bennett’s focus on vision, and Daniel Wright’s balancing role are presented as complementary rather than mutually exclusive.
EOF
```

## Step 5 — Ingest script (embeddings → Qdrant)

Create ingest.py:
```python
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "mentions_mvp")

TRANSCRIPT_PATH = os.environ.get("TRANSCRIPT_PATH", "data/transcripts/sample.txt")

def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    path = Path(TRANSCRIPT_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")

    raw_text = path.read_text(encoding="utf-8")

    # 1) Split text into chunks (simple MVP)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    docs = splitter.create_documents([raw_text])

    # 2) Attach minimal metadata (manual placeholders for now)
    video_url = os.environ.get("VIDEO_URL", "https://youtube.com/watch?v=VIDEO_ID")
    for i, d in enumerate(docs):
        d.metadata.update({
            "source_file": str(path),
            "chunk_id": i,
            "video_url": video_url,
            # placeholders; in later phases set real timestamps per chunk
            "start_sec": None,
            "end_sec": None,
        })

    # 3) Create embeddings + store in Qdrant
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION,
    )

    print(f"✅ Ingested {len(docs)} chunks into Qdrant collection: {COLLECTION}")
    print(f"Qdrant URL: {QDRANT_URL}")

if __name__ == "__main__":
    main()
```

Run ingest:

```
uv run python ingest.py
```

## Step 6 — Query script (prompt → retrieve → print)

Create query.py:


```
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "mentions_mvp")

def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    query = os.environ.get("QUERY", "What is Michael Harris's approach to leadership?")
    k = int(os.environ.get("TOP_K", "3"))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION,
    )

    results = store.similarity_search_with_score(query, k=k)

    print(f"\n🔎 Query: {query}")
    print(f"Top {k} results:\n")

    for rank, (doc, score) in enumerate(results, start=1):
        md = doc.metadata
        print(f"#{rank}  score={score:.4f}")
        print(f"   video_url: {md.get('video_url')}")
        print(f"   source_file: {md.get('source_file')}")
        print(f"   chunk_id: {md.get('chunk_id')}")
        print(f"   start_sec: {md.get('start_sec')}  end_sec: {md.get('end_sec')}")
        print("   text:")
        print("   " + doc.page_content.strip().replace("\n", "\n   "))
        print()

if __name__ == "__main__":
    main()
```

Run query:

```
export QUERY="What does Laura Bennett think about organizational change?"
export TOP_K=3
uv run python query.py
```

## What “works” after Phase 1

- You can ingest a transcript text into Qdrant
- You can query in natural language and retrieve relevant chunks
- You can print metadata + text to console (foundation for building YouTube links)
