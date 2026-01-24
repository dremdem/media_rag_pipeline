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

Two-stage pipeline to extract opinions about persons from transcript chunks:

```mermaid
graph LR
    A[Transcript Chunks] -->|Every chunk| B[NER Service]
    B -->|Has PERSON?| C{Filter}
    C -->|No| D[Skip]
    C -->|Yes| E[LLM Opinion Extraction]
    E --> F[Store Opinion + Persona]
```

**Why two stages?**

| Stage | Tool | Cost | Speed | Purpose |
|-------|------|------|-------|---------|
| 1 | NER Service | Free (local) | ~50ms | Detect if chunk mentions any person |
| 2 | LLM (GPT-4) | ~$0.01-0.03 | ~2s | Extract opinion about person |

**Key distinction:**
- **Mention** (NER): "Иванов встретился с Кузнецовым" → persons found, no opinion
- **Opinion** (LLM): "Сидоров назвал Иванова некомпетентным" → opinion about Иванов by Сидоров

This reduces LLM calls by 30-70% by filtering chunks without person mentions.

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

See [services/ner/README.md](../services/ner/README.md) for details.

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
│   └── ner/             # Russian PERSON-NER microservice
│       ├── app/
│       │   └── main.py  # FastAPI application
│       ├── Dockerfile
│       └── README.md
├── data/
│   └── transcripts/     # Transcripts, SRT, audio
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
