"""Pydantic models for LLM Analyzer Service."""

from typing import Literal

from pydantic import BaseModel, Field

# --- Common Types ---

Polarity = Literal["negative", "positive", "mixed", "unclear"]
SegmentType = Literal["narrative", "qa"]


# --- Opinion Detection Models ---


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


# --- Q&A Segmentation Models ---


class Utterance(BaseModel):
    """Single utterance from Deepgram output."""

    u: int = Field(..., ge=0, description="Utterance index")
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    text: str = Field(..., description="Utterance text")


class BoundarySegment(BaseModel):
    """A single boundary segment (narrative or Q&A)."""

    type: SegmentType = Field(description="Segment type: 'narrative' or 'qa'")
    start_u: int = Field(..., ge=0, description="Start utterance index")
    end_u: int = Field(..., ge=0, description="End utterance index (inclusive)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    notes: str | None = Field(default=None, description="Optional notes about the segment")


class BoundaryRequest(BaseModel):
    """Request for Pass 1: boundary detection."""

    video_id: str = Field(..., min_length=1, description="Video identifier")
    utterances: list[Utterance] = Field(..., min_length=1, description="List of utterances")


class BoundaryResponse(BaseModel):
    """Response for Pass 1: boundary segments."""

    video_id: str
    segments: list[BoundarySegment]


class QARange(BaseModel):
    """Range of utterance indices for Q&A region."""

    start_u: int = Field(..., ge=0, description="Start utterance index")
    end_u: int = Field(..., ge=0, description="End utterance index (inclusive)")


class QABlock(BaseModel):
    """A single semantic Q&A block."""

    start_u: int = Field(..., ge=0, description="Start utterance index")
    end_u: int = Field(..., ge=0, description="End utterance index (inclusive)")
    questions: list[str] = Field(default_factory=list, description="Detected questions (if any)")
    answer_summary: str = Field(..., description="Short summary of the answer")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")


class BlocksRequest(BaseModel):
    """Request for Pass 2: Q&A block segmentation."""

    video_id: str = Field(..., min_length=1, description="Video identifier")
    utterances: list[Utterance] = Field(..., min_length=1, description="Q&A region utterances only")
    qa_range: QARange = Field(..., description="Original Q&A range for validation")


class BlocksResponse(BaseModel):
    """Response for Pass 2: Q&A blocks."""

    video_id: str
    qa_blocks: list[QABlock]


class FullSegmentRequest(BaseModel):
    """Request for combined Pass 1 + Pass 2 segmentation."""

    video_id: str = Field(..., min_length=1, description="Video identifier")
    utterances: list[Utterance] = Field(..., min_length=1, description="All utterances")


class DeepgramSegmentRequest(BaseModel):
    """Request for segmentation from raw Deepgram JSON output.

    Accepts the full Deepgram transcription JSON file, extracts utterances,
    and runs the full segmentation pipeline.
    """

    video_id: str = Field(..., min_length=1, description="Video identifier")
    deepgram_json: dict = Field(..., description="Raw Deepgram API response JSON")


class StoredBoundarySegment(BaseModel):
    """Stored boundary segment with timestamps."""

    seg_id: str
    type: SegmentType
    start_u: int
    end_u: int
    start: float
    end: float
    confidence: float
    notes: str | None
    created_at: str


class StoredQABlock(BaseModel):
    """Stored Q&A block with timestamps and assembled text."""

    block_id: str
    start_u: int
    end_u: int
    start: float
    end: float
    questions: list[str]
    answer_summary: str
    confidence: float
    text: str = Field(..., description="Assembled text from utterances")
    created_at: str


class SegmentsResponse(BaseModel):
    """Response for GET /segments/{video_id}."""

    video_id: str
    boundary_segments: list[StoredBoundarySegment]
    qa_blocks: list[StoredQABlock]


class ExportData(BaseModel):
    """Full export structure for a video."""

    video_id: str
    boundary_segments: list[dict]
    qa_blocks: list[dict]
    created_at: str


class ExportResponse(BaseModel):
    """Response for GET /exports/{video_id}."""

    video_id: str
    export: ExportData
    file_path: str | None = Field(default=None, description="Path to exported JSON file")


# --- Health Check ---


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str
    model: str
    version: str
