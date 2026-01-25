"""Pydantic models for Opinion Detector Service."""

from typing import Literal

from pydantic import BaseModel, Field

Polarity = Literal["negative", "positive", "mixed", "unclear"]


class DetectRequest(BaseModel):
    """Request model for single opinion detection."""

    chunk_id: str = Field(..., min_length=1, description="Unique chunk identifier")
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    text: str = Field(..., min_length=1, description="Chunk text to analyze")
    persons: list[str] = Field(default_factory=list, description="List of persons from NER")


class DetectResponse(BaseModel):
    """Response model for opinion detection."""

    has_opinion: bool = Field(description="Whether text contains opinion about any person")
    targets: list[str] = Field(description="Persons who are targets of opinions")
    opinion_spans: list[str] = Field(description="Direct quotes containing opinions")
    polarity: Polarity = Field(description="Overall opinion polarity")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")


class DetectBatchRequest(BaseModel):
    """Request model for batch opinion detection."""

    items: list[DetectRequest] = Field(..., min_length=1, description="List of chunks to analyze")


class DetectBatchResponse(BaseModel):
    """Response model for batch opinion detection."""

    results: list[DetectResponse]
    total_with_opinions: int = Field(description="Count of chunks with opinions")


class ChunkResponse(BaseModel):
    """Response model for stored chunk retrieval."""

    chunk_id: str
    start: float
    end: float
    persons: list[str]
    has_opinion: bool
    targets: list[str]
    opinion_spans: list[str]
    polarity: Polarity
    confidence: float
    created_at: str


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str
    model: str
    version: str
