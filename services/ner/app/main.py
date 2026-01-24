"""Russian PERSON-NER Service.

A lightweight FastAPI service for detecting person mentions in Russian text.
Uses r1char9/ner-rubert-tiny-news model, optimized for news/media content.

This service acts as a cheap filter in the opinion extraction pipeline:
1. NER detects if chunk mentions any persons (fast, free)
2. Only chunks with persons are sent to LLM for opinion extraction (slow, expensive)
"""

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

MODEL_ID = "r1char9/ner-rubert-tiny-news"

app = FastAPI(
    title="ru-person-ner",
    version="1.0.0",
    description="Lightweight Russian NER service for detecting person mentions",
)

# Load model once at startup
ner = pipeline(
    task="ner",
    model=MODEL_ID,
    aggregation_strategy="simple",  # merges sub-tokens into full entities
)


class NerRequest(BaseModel):
    """Request model for NER endpoint."""

    text: str = Field(..., min_length=1, description="Chunk of Russian text")
    return_raw: bool = Field(False, description="Return raw NER spans as well")


class NerBatchRequest(BaseModel):
    """Request model for batch NER endpoint."""

    texts: list[str] = Field(..., min_length=1, description="List of Russian text chunks")
    return_raw: bool = Field(False, description="Return raw NER spans as well")


class NerResponse(BaseModel):
    """Response model for NER endpoint."""

    persons: list[str]
    has_persons: bool = Field(description="Quick check if any persons were found")
    raw: list[dict[str, Any]] | None = None


class NerBatchResponse(BaseModel):
    """Response model for batch NER endpoint."""

    results: list[NerResponse]
    total_with_persons: int = Field(description="Count of texts that have persons")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str
    model: str
    version: str


def _is_person(ent: dict[str, Any]) -> bool:
    """Check if entity is a person."""
    label = (ent.get("entity_group") or ent.get("entity") or "").upper()
    return "PER" in label or "PERSON" in label


def _extract_persons(text: str, return_raw: bool = False) -> NerResponse:
    """Extract person entities from a single text.

    Args:
        text: Russian text to analyze
        return_raw: Whether to include raw NER spans

    Returns:
        NerResponse with extracted persons
    """
    ents = ner(text)

    persons: list[str] = []
    for e in ents:
        if _is_person(e):
            word = (e.get("word") or "").strip()
            if word:
                persons.append(word)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in persons:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return NerResponse(
        persons=unique,
        has_persons=len(unique) > 0,
        raw=ents if return_raw else None,
    )


@app.get("/healthz", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint for Docker orchestration."""
    return HealthResponse(
        status="healthy",
        model=MODEL_ID,
        version=app.version,
    )


@app.post("/ner/persons", response_model=NerResponse)
def ner_persons(req: NerRequest) -> NerResponse:
    """Extract person entities from Russian text.

    This endpoint detects mentions of people in the input text.
    Use it as a filter before calling expensive LLM for opinion extraction.

    Example:
        Input: "Иванов раскритиковал Петрова, а Кузнецова похвалил."
        Output: {"persons": ["Иванов", "Петрова", "Кузнецова"], "has_persons": true}
    """
    return _extract_persons(req.text, req.return_raw)


@app.post("/ner/persons/batch", response_model=NerBatchResponse)
def ner_persons_batch(req: NerBatchRequest) -> NerBatchResponse:
    """Extract person entities from multiple texts.

    This endpoint processes a batch of texts and returns results for each.
    More efficient than calling /ner/persons multiple times.

    Example:
        Input: {"texts": ["Иванов met Петров.", "No persons here."], "return_raw": false}
        Output: {"results": [...], "total_with_persons": 1}
    """
    results = [_extract_persons(text, req.return_raw) for text in req.texts]
    total_with_persons = sum(1 for r in results if r.has_persons)

    return NerBatchResponse(
        results=results,
        total_with_persons=total_with_persons,
    )
