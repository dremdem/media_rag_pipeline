"""Opinion Detector Service.

A FastAPI service that detects whether text contains opinions about persons.
Uses OpenAI API for classification and stores results in SQLite.

This service is step 3 in the opinion extraction pipeline:
1. Deepgram -> transcript chunks
2. NER service -> persons[] per chunk
3. Opinion Detector (this service) -> has_opinion, targets, spans
4. (Optional) Opinion Extractor -> structured extraction
"""

import json
import logging
import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .db import get_detection, init_db, upsert_detection
from .schemas import (
    ChunkResponse,
    DetectBatchRequest,
    DetectBatchResponse,
    DetectRequest,
    DetectResponse,
    HealthResponse,
)

# --- Configuration ---
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", "4000"))

# Validate API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is required. "
        "Get yours at: https://platform.openai.com/api-keys"
    )

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="opinion-detector",
    version="1.0.0",
    description="Detects opinions about persons in Russian text using OpenAI",
)

client = OpenAI(api_key=OPENAI_API_KEY)


@app.on_event("startup")
def _startup() -> None:
    """Initialize database on startup."""
    init_db()
    logger.info(f"Opinion Detector started with model: {OPENAI_MODEL}")


def _build_prompt(req: DetectRequest) -> str:
    """Build the prompt for OpenAI."""
    payload = {
        "task": "Detect whether the author expresses an opinion about any PERSON in the text.",
        "language": "ru",
        "definitions": {
            "opinion": (
                "Any evaluative judgment, praise/blame, accusation, sarcasm/irony, "
                "attribution of motives/intentions, predictions about a person, or conclusions about a person. "
                "Pure factual mentions are NOT opinions."
            )
        },
        "input": {"text": req.text, "persons": req.persons},
        "output_schema": {
            "has_opinion": "boolean",
            "targets": "array of strings (subset of persons)",
            "opinion_spans": "array of short direct quotes copied from input text",
            "polarity": "one of: negative, positive, mixed, unclear",
            "confidence": "number 0..1",
        },
        "rules": [
            "Return ONLY valid JSON. No extra text.",
            "targets MUST be chosen only from provided persons list.",
            "If has_opinion=false then targets must be empty and opinion_spans must be empty.",
            "opinion_spans MUST be exact substrings from the input text (copy-paste).",
            "If the text contains sarcasm or irony toward a person, mark has_opinion=true.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _call_openai(prompt: str) -> DetectResponse:
    """Call OpenAI API with retry logic."""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a strict information extraction engine. Return ONLY valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,  # Low temperature for consistent extraction
    )

    text_out = resp.choices[0].message.content or ""

    # Log token usage for cost monitoring
    if resp.usage:
        logger.debug(
            f"Tokens used: {resp.usage.total_tokens} "
            f"(prompt: {resp.usage.prompt_tokens}, completion: {resp.usage.completion_tokens})"
        )

    # Parse JSON
    try:
        data = json.loads(text_out)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Model returned invalid JSON: {text_out[:500]}. Error: {e}",
        )

    return DetectResponse(**data)


def _detect_single(req: DetectRequest) -> DetectResponse:
    """Process a single detection request."""
    # If no persons, skip OpenAI call
    if not req.persons:
        return DetectResponse(
            has_opinion=False,
            targets=[],
            opinion_spans=[],
            polarity="unclear",
            confidence=1.0,
        )

    # Truncate text if too long
    text = req.text
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH] + "..."
        req = DetectRequest(
            chunk_id=req.chunk_id,
            start=req.start,
            end=req.end,
            text=text,
            persons=req.persons,
        )
        logger.warning(f"Truncated text for chunk {req.chunk_id} to {MAX_TEXT_LENGTH} chars")

    # Call OpenAI
    prompt = _build_prompt(req)
    try:
        result = _call_openai(prompt)
    except Exception as e:
        logger.error(f"OpenAI API error for chunk {req.chunk_id}: {e}")
        # Graceful degradation: return safe default
        return DetectResponse(
            has_opinion=False,
            targets=[],
            opinion_spans=[],
            polarity="unclear",
            confidence=0.0,
        )

    # Validate targets are from persons list
    invalid_targets = [t for t in result.targets if t not in req.persons]
    if invalid_targets:
        logger.warning(f"Model returned invalid targets: {invalid_targets}")
        result.targets = [t for t in result.targets if t in req.persons]

    # Validate spans are in text (log warning but don't fail)
    for span in result.opinion_spans:
        if span and span not in req.text:
            logger.warning(f"Span not found in text: {span[:50]}...")

    return result


def _persist_result(req: DetectRequest, result: DetectResponse) -> None:
    """Persist detection result to SQLite."""
    now = datetime.now(timezone.utc).isoformat()
    upsert_detection(
        chunk_id=req.chunk_id,
        start=req.start,
        end=req.end,
        persons_json=json.dumps(req.persons, ensure_ascii=False),
        has_opinion=1 if result.has_opinion else 0,
        targets_json=json.dumps(result.targets, ensure_ascii=False),
        spans_json=json.dumps(result.opinion_spans, ensure_ascii=False),
        polarity=result.polarity,
        confidence=float(result.confidence),
        created_at=now,
    )


@app.post("/detect-opinion", response_model=DetectResponse)
def detect_opinion(req: DetectRequest) -> DetectResponse:
    """Detect opinion about persons in a single text chunk.

    Example:
        Input: {"chunk_id": "demo_001", "start": 0, "end": 30,
                "text": "Вот такое стремление Иванова к миру.",
                "persons": ["Иванов"]}
        Output: {"has_opinion": true, "targets": ["Иванов"], ...}
    """
    result = _detect_single(req)
    _persist_result(req, result)
    return result


@app.post("/detect-opinion/batch", response_model=DetectBatchResponse)
def detect_opinion_batch(req: DetectBatchRequest) -> DetectBatchResponse:
    """Detect opinions in multiple text chunks.

    More efficient than calling /detect-opinion multiple times.
    Results are persisted to SQLite for each chunk.
    """
    results: list[DetectResponse] = []
    for item in req.items:
        result = _detect_single(item)
        _persist_result(item, result)
        results.append(result)

    total_with_opinions = sum(1 for r in results if r.has_opinion)

    return DetectBatchResponse(
        results=results,
        total_with_opinions=total_with_opinions,
    )


@app.get("/chunks/{chunk_id}", response_model=ChunkResponse)
def read_chunk(chunk_id: str) -> ChunkResponse:
    """Retrieve stored detection result by chunk_id."""
    row = get_detection(chunk_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Chunk not found: {chunk_id}")

    return ChunkResponse(
        chunk_id=row["chunk_id"],
        start=row["start"],
        end=row["end"],
        persons=json.loads(row["persons_json"]),
        has_opinion=row["has_opinion"],
        targets=json.loads(row["targets_json"]),
        opinion_spans=json.loads(row["spans_json"]),
        polarity=row["polarity"],
        confidence=row["confidence"],
        created_at=row["created_at"],
    )


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Health check endpoint for Docker orchestration."""
    return HealthResponse(
        status="healthy",
        model=OPENAI_MODEL,
        version=app.version,
    )
