# Architecture

> High-level system architecture for the Media RAG Pipeline.

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
  - [Data Ingestion](#data-ingestion)
  - [Query Flow](#query-flow)
  - [Opinion Extraction](#opinion-extraction)
- [Components](#components)
  - [Transcription](#transcription)
  - [NER Service](#ner-service)
  - [Opinion Detector Service](#opinion-detector-service)
  - [Embeddings](#embeddings)
  - [Vector Database](#vector-database)
  - [Orchestration](#orchestration)

---

## Overview

The Media RAG Pipeline extracts transcripts from YouTube videos, generates embeddings, stores them in a vector database, and enables semantic search.

```mermaid
graph LR
    A[YouTube Video] -->|Deepgram| B[Transcript]
    B -->|OpenAI| C[Embeddings]
    C -->|Store| D[(Qdrant)]
    E[User Query] -->|OpenAI| F[Query Embedding]
    F -->|Search| D
    D -->|Results| G[Relevant Chunks]
```

---

## Pipeline

### Data Ingestion

```mermaid
graph TD
    A[YouTube URL] -->|yt-dlp| B[Audio File]
    B -->|Deepgram API| C[Transcript + Timestamps]
    C -->|Text Splitter| D[Chunks]
    D -->|OpenAI Embeddings| E[Vectors]
    E -->|Store| F[(Qdrant Collection)]
```

**Steps:**
1. Download audio from YouTube using `yt-dlp`
2. Transcribe with Deepgram (word-level timestamps)
3. Split transcript into chunks (800 chars, 120 overlap)
4. Generate embeddings via OpenAI API
5. Store vectors with metadata in Qdrant

### Query Flow

```mermaid
graph TD
    A[User Query] -->|OpenAI Embeddings| B[Query Vector]
    B -->|Similarity Search| C[(Qdrant)]
    C -->|Top-K Results| D[Relevant Chunks]
    D -->|Return| E[Text + Metadata]
```

**Steps:**
1. Convert user query to embedding
2. Search Qdrant for similar vectors
3. Return top-k chunks with metadata (timestamps, source)

### Opinion Extraction

Three-stage pipeline to extract opinions about persons from transcript chunks:

```mermaid
graph LR
    A[Transcript Chunks] -->|Every chunk| B[NER Service]
    B -->|Has PERSON?| C{Filter}
    C -->|No| D[Skip]
    C -->|Yes| E[Opinion Detector]
    E -->|has_opinion?| F{Filter}
    F -->|No| G[Skip]
    F -->|Yes| H[Store Result]
```

**Pipeline stages:**

| Stage | Service | Port | Cost | Speed | Purpose |
|-------|---------|------|------|-------|---------|
| 1 | NER Service | 8000 | Free (local) | ~50ms | Detect person mentions |
| 2 | Opinion Detector | 8001 | ~$0.001/chunk | ~1s | Detect if opinion exists |
| 3 | (Optional) Extractor | - | ~$0.01/chunk | ~2s | Deep structured extraction |

**Key distinction:**
- **Mention** (NER): "Иванов встретился с Кузнецовым" → persons found, no opinion
- **Opinion** (Detector): "Сидоров назвал Иванова некомпетентным" → opinion about Иванов

This reduces costs by 50-80% through progressive filtering.

---

## Components

### Transcription

| Service | Purpose | Status |
|---------|---------|--------|
| **Deepgram** | Speech-to-text transcription | Active |
| yt-dlp | YouTube audio download | Active |

**Deepgram Features Used:**
- Model: `nova-3` (latest)
- Smart formatting (dates, numbers, emails)
- Word-level timestamps
- SRT subtitle generation
- Optional speaker diarization

### NER Service

| Service | Model | Purpose | Status |
|---------|-------|---------|--------|
| **ru-person-ner** | `r1char9/ner-rubert-tiny-news` | Detect person mentions | Active |

**Features:**
- CPU-only (no GPU required)
- ~50ms latency per chunk
- Docker deployment
- Health check endpoint (`/healthz`)

**API Endpoint:**
```
POST /ner/persons
{
  "text": "Иванов раскритиковал Петрова.",
  "return_raw": false
}
→ {"persons": ["Иванов", "Петрова"], "has_persons": true}
```

See [NER_SERVICE.md](./NER_SERVICE.md) for details.

### Opinion Detector Service

| Service | Model | Purpose | Status |
|---------|-------|---------|--------|
| **opinion-detector** | `gpt-4o-mini` | Detect opinions about persons | Active |

**Features:**
- OpenAI-powered classification
- SQLite persistence for caching
- Batch endpoint for efficiency
- ~$0.001/chunk cost

**API Endpoints:**
```
POST /detect-opinion      # Single chunk
POST /detect-opinion/batch # Multiple chunks
GET  /chunks/{chunk_id}   # Retrieve stored result
GET  /healthz             # Health check
```

See [OPINION_DETECTOR_SERVICE.md](./OPINION_DETECTOR_SERVICE.md) for details.

### Embeddings

| Provider | Model | Dimensions | Status |
|----------|-------|------------|--------|
| **OpenAI** | `text-embedding-3-small` | 1,536 | Active |
| OpenAI | `text-embedding-3-large` | 3,072 | Available |
| Cohere | Embed API | - | Planned |
| Hugging Face | Sentence-Transformers | - | Planned |

### Vector Database

| Database | Deployment | Status |
|----------|------------|--------|
| **Qdrant** | Docker (local) | Active |
| Chroma | - | Planned |

**Qdrant Configuration:**
- Collection: `mentions_mvp`
- Distance metric: Cosine similarity
- Ports: 6333 (REST), 6334 (gRPC)

### Orchestration

| Tool | Purpose | Status |
|------|---------|--------|
| **LangChain** | Pipeline orchestration | Active |
| python-dotenv | Environment management | Active |

---

## Directory Structure

```
media_rag_pipeline/
├── src/
│   ├── ingest.py        # Ingest transcripts → Qdrant
│   ├── query.py         # Query Qdrant
│   └── transcribe.py    # YouTube → Deepgram → files
├── services/
│   ├── ner/             # Russian PERSON-NER microservice (port 8000)
│   │   ├── app/
│   │   │   └── main.py
│   │   └── Dockerfile
│   └── opinion-detector/ # Opinion detection service (port 8001)
│       ├── app/
│       │   ├── main.py
│       │   ├── schemas.py
│       │   └── db.py
│       └── Dockerfile
├── data/
│   ├── transcripts/     # Transcripts, SRT, audio
│   └── opinions/        # SQLite database (gitignored)
├── docs/                # Documentation
└── qdrant_storage/      # Qdrant data (gitignored)
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API for embeddings |
| `DEEPGRAM_API_KEY` | Yes | Deepgram API for transcription |
| `QDRANT_URL` | No | Qdrant URL (default: localhost:6333) |
| `QDRANT_COLLECTION` | No | Collection name (default: mentions_mvp) |
