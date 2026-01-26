"""LLM Analyzer Service.

A FastAPI service that provides LLM-powered text analysis:
1. Opinion detection - detects opinions about persons in text
2. Q&A segmentation - segments transcripts into narrative/Q&A regions and semantic blocks

Pipeline position:
- Deepgram -> transcript/utterances
- NER service -> persons[] per chunk
- LLM Analyzer (this service) -> opinions, segments, blocks
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .db import (
    delete_blocks,
    delete_boundaries,
    get_blocks,
    get_boundaries,
    get_detection,
    get_export,
    init_db,
    upsert_block,
    upsert_boundary,
    upsert_detection,
    upsert_export,
)
from .prompts import (
    SYSTEM_PROMPT_EXTRACTION,
    SYSTEM_PROMPT_SEGMENTATION,
    build_blocks_prompt,
    build_boundary_prompt,
    build_opinion_prompt,
)
from .schemas import (
    BlocksRequest,
    BlocksResponse,
    BoundaryRequest,
    BoundaryResponse,
    BoundarySegment,
    ChunkResponse,
    DeepgramSegmentRequest,
    DetectBatchRequest,
    DetectBatchResponse,
    DetectRequest,
    DetectResponse,
    ExportData,
    ExportResponse,
    FullSegmentRequest,
    HealthResponse,
    QABlock,
    SegmentsResponse,
    StoredBoundarySegment,
    StoredQABlock,
    Utterance,
)

# --- Configuration ---
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
# Use stronger model for complex block segmentation (Pass 2)
# Block segmentation has ~50k tokens and requires better instruction following
OPENAI_MODEL_BLOCKS = os.environ.get("OPENAI_MODEL_BLOCKS", "gpt-4o")
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", "4000"))
EXPORTS_DIR = Path(os.environ.get("EXPORTS_DIR", "exports"))

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
    title="llm-analyzer",
    version="2.0.0",
    description="LLM-powered text analysis: opinion detection and Q&A segmentation",
)

client = OpenAI(api_key=OPENAI_API_KEY)


@app.on_event("startup")
def _startup() -> None:
    """Initialize database and exports directory on startup."""
    init_db()
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"LLM Analyzer started with model: {OPENAI_MODEL}")
    logger.info(f"Block segmentation model: {OPENAI_MODEL_BLOCKS}")


# --- OpenAI Helpers ---


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _call_openai_json(
    prompt: str,
    system_prompt: str,
    model: str | None = None,
) -> dict:
    """Call OpenAI API with retry logic, returns parsed JSON dict.

    Args:
        prompt: User prompt to send
        system_prompt: System prompt for the model
        model: Optional model override. If None, uses OPENAI_MODEL default.
    """
    use_model = model or OPENAI_MODEL
    logger.info(f"Calling OpenAI with model: {use_model}")

    resp = client.chat.completions.create(
        model=use_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    text_out = resp.choices[0].message.content or ""

    if resp.usage:
        logger.debug(
            f"Tokens used: {resp.usage.total_tokens} "
            f"(prompt: {resp.usage.prompt_tokens}, completion: {resp.usage.completion_tokens})"
        )

    try:
        return json.loads(text_out)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Model returned invalid JSON: {text_out[:500]}. Error: {e}",
        )


# =============================================================================
# OPINION DETECTION ENDPOINTS
# =============================================================================


def _detect_single(req: DetectRequest) -> DetectResponse:
    """Process a single opinion detection request."""
    if not req.persons:
        return DetectResponse(
            has_opinion=False,
            targets=[],
            opinion_spans=[],
            polarity="unclear",
            confidence=1.0,
        )

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

    prompt = build_opinion_prompt(req)
    try:
        data = _call_openai_json(prompt, SYSTEM_PROMPT_EXTRACTION)
        result = DetectResponse(**data)
    except Exception as e:
        logger.error(f"OpenAI API error for chunk {req.chunk_id}: {e}")
        return DetectResponse(
            has_opinion=False,
            targets=[],
            opinion_spans=[],
            polarity="unclear",
            confidence=0.0,
        )

    invalid_targets = [t for t in result.targets if t not in req.persons]
    if invalid_targets:
        logger.warning(f"Model returned invalid targets: {invalid_targets}")
        result.targets = [t for t in result.targets if t in req.persons]

    for span in result.opinion_spans:
        if span and span not in req.text:
            logger.warning(f"Span not found in text: {span[:50]}...")

    return result


def _persist_detection(req: DetectRequest, result: DetectResponse) -> None:
    """Persist opinion detection result to SQLite."""
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
    """Detect opinion about persons in a single text chunk."""
    result = _detect_single(req)
    _persist_detection(req, result)
    return result


@app.post("/detect-opinion/batch", response_model=DetectBatchResponse)
def detect_opinion_batch(req: DetectBatchRequest) -> DetectBatchResponse:
    """Detect opinions in multiple text chunks."""
    results: list[DetectResponse] = []
    for item in req.items:
        result = _detect_single(item)
        _persist_detection(item, result)
        results.append(result)

    return DetectBatchResponse(
        results=results,
        total_with_opinions=sum(1 for r in results if r.has_opinion),
    )


@app.get("/chunks/{chunk_id}", response_model=ChunkResponse)
def read_chunk(chunk_id: str) -> ChunkResponse:
    """Retrieve stored opinion detection result by chunk_id."""
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


# =============================================================================
# Q&A SEGMENTATION ENDPOINTS
# =============================================================================


def _validate_segments(segments: list[dict], max_u: int) -> list[BoundarySegment]:
    """Validate and parse boundary segments from LLM output."""
    validated = []
    for seg in segments:
        start_u = seg.get("start_u", 0)
        end_u = seg.get("end_u", 0)

        if not (0 <= start_u <= end_u <= max_u):
            logger.warning(f"Invalid segment indices: start_u={start_u}, end_u={end_u}, max={max_u}")
            continue

        seg_type = seg.get("type", "narrative")
        if seg_type not in ("narrative", "qa"):
            seg_type = "narrative"

        confidence = min(1.0, max(0.0, float(seg.get("confidence", 0.5))))

        validated.append(
            BoundarySegment(
                type=seg_type,
                start_u=start_u,
                end_u=end_u,
                confidence=confidence,
                notes=seg.get("notes"),
            )
        )

    return validated


def _validate_blocks(blocks: list[dict], qa_start: int, qa_end: int) -> list[QABlock]:
    """Validate and parse Q&A blocks from LLM output."""
    validated = []
    for block in blocks:
        start_u = block.get("start_u", qa_start)
        end_u = block.get("end_u", qa_end)

        if not (qa_start <= start_u <= end_u <= qa_end):
            logger.warning(
                f"Invalid block indices: start_u={start_u}, end_u={end_u}, "
                f"qa_range=[{qa_start}, {qa_end}]"
            )
            continue

        confidence = min(1.0, max(0.0, float(block.get("confidence", 0.5))))
        questions = block.get("questions", [])
        if not isinstance(questions, list):
            questions = []

        validated.append(
            QABlock(
                start_u=start_u,
                end_u=end_u,
                questions=questions,
                answer_summary=block.get("answer_summary", ""),
                confidence=confidence,
            )
        )

    return validated


def _get_utterance_by_index(utterances: list[Utterance], idx: int) -> Utterance | None:
    """Find utterance by its index (u field)."""
    for u in utterances:
        if u.u == idx:
            return u
    return None


def _assemble_text(utterances: list[Utterance], start_u: int, end_u: int) -> str:
    """Assemble text from utterances in range [start_u, end_u]."""
    texts = []
    for u in utterances:
        if start_u <= u.u <= end_u:
            texts.append(u.text)
    return " ".join(texts)


@app.post("/segment/qa/boundaries", response_model=BoundaryResponse)
def segment_boundaries(req: BoundaryRequest) -> BoundaryResponse:
    """Pass 1: Segment transcript into narrative vs Q&A regions.

    Takes utterances and returns boundary segments with utterance index ranges.
    Results are persisted to SQLite.
    """
    if not req.utterances:
        raise HTTPException(status_code=400, detail="No utterances provided")

    max_u = max(u.u for u in req.utterances)

    prompt = build_boundary_prompt(req.utterances)
    try:
        data = _call_openai_json(prompt, SYSTEM_PROMPT_SEGMENTATION)
    except Exception as e:
        logger.error(f"OpenAI API error for boundary segmentation: {e}")
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    raw_segments = data.get("segments", [])
    segments = _validate_segments(raw_segments, max_u)

    if not segments:
        segments = [
            BoundarySegment(
                type="narrative",
                start_u=0,
                end_u=max_u,
                confidence=0.5,
                notes="Fallback: entire transcript as narrative",
            )
        ]

    delete_boundaries(req.video_id)
    now = datetime.now(timezone.utc).isoformat()

    for i, seg in enumerate(segments):
        start_utt = _get_utterance_by_index(req.utterances, seg.start_u)
        end_utt = _get_utterance_by_index(req.utterances, seg.end_u)

        upsert_boundary(
            video_id=req.video_id,
            seg_id=f"seg_{i:03d}",
            seg_type=seg.type,
            start_u=seg.start_u,
            end_u=seg.end_u,
            start=start_utt.start if start_utt else 0.0,
            end=end_utt.end if end_utt else 0.0,
            confidence=seg.confidence,
            notes=seg.notes,
            created_at=now,
        )

    return BoundaryResponse(video_id=req.video_id, segments=segments)


@app.post("/segment/qa/blocks", response_model=BlocksResponse)
def segment_blocks(req: BlocksRequest) -> BlocksResponse:
    """Pass 2: Segment Q&A region into semantic answer blocks.

    Takes Q&A region utterances and returns semantic blocks.
    Results are persisted to SQLite.
    """
    if not req.utterances:
        raise HTTPException(status_code=400, detail="No utterances provided")

    qa_range = {"start_u": req.qa_range.start_u, "end_u": req.qa_range.end_u}

    prompt = build_blocks_prompt(req.utterances, qa_range)
    try:
        # Use stronger model for block segmentation (complex task with ~50k tokens)
        data = _call_openai_json(prompt, SYSTEM_PROMPT_SEGMENTATION, model=OPENAI_MODEL_BLOCKS)
    except Exception as e:
        logger.error(f"OpenAI API error for block segmentation: {e}")
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    raw_blocks = data.get("qa_blocks", [])
    blocks = _validate_blocks(raw_blocks, req.qa_range.start_u, req.qa_range.end_u)

    if not blocks:
        blocks = [
            QABlock(
                start_u=req.qa_range.start_u,
                end_u=req.qa_range.end_u,
                questions=[],
                answer_summary="Fallback: entire Q&A region as one block",
                confidence=0.5,
            )
        ]

    delete_blocks(req.video_id)
    now = datetime.now(timezone.utc).isoformat()

    for i, block in enumerate(blocks):
        start_utt = _get_utterance_by_index(req.utterances, block.start_u)
        end_utt = _get_utterance_by_index(req.utterances, block.end_u)

        upsert_block(
            video_id=req.video_id,
            block_id=f"qa_{i:03d}",
            start_u=block.start_u,
            end_u=block.end_u,
            start=start_utt.start if start_utt else 0.0,
            end=end_utt.end if end_utt else 0.0,
            questions_json=json.dumps(block.questions, ensure_ascii=False),
            answer_summary=block.answer_summary,
            confidence=block.confidence,
            created_at=now,
        )

    return BlocksResponse(video_id=req.video_id, qa_blocks=blocks)


@app.post("/segment/qa/run", response_model=SegmentsResponse)
def segment_full(req: FullSegmentRequest) -> SegmentsResponse:
    """Combined Pass 1 + Pass 2: Full Q&A segmentation pipeline.

    Runs boundary detection, then for each Q&A region, runs block segmentation.
    Returns complete segmentation results.
    """
    boundary_req = BoundaryRequest(video_id=req.video_id, utterances=req.utterances)
    boundary_resp = segment_boundaries(boundary_req)

    all_blocks: list[StoredQABlock] = []
    block_counter = 0

    for seg in boundary_resp.segments:
        if seg.type != "qa":
            continue

        qa_utterances = [u for u in req.utterances if seg.start_u <= u.u <= seg.end_u]
        if not qa_utterances:
            continue

        blocks_req = BlocksRequest(
            video_id=req.video_id,
            utterances=qa_utterances,
            qa_range={"start_u": seg.start_u, "end_u": seg.end_u},
        )

        try:
            blocks_resp = segment_blocks(blocks_req)

            for block in blocks_resp.qa_blocks:
                start_utt = _get_utterance_by_index(req.utterances, block.start_u)
                end_utt = _get_utterance_by_index(req.utterances, block.end_u)
                text = _assemble_text(req.utterances, block.start_u, block.end_u)

                all_blocks.append(
                    StoredQABlock(
                        block_id=f"qa_{block_counter:03d}",
                        start_u=block.start_u,
                        end_u=block.end_u,
                        start=start_utt.start if start_utt else 0.0,
                        end=end_utt.end if end_utt else 0.0,
                        questions=block.questions,
                        answer_summary=block.answer_summary,
                        confidence=block.confidence,
                        text=text,
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                )
                block_counter += 1

        except Exception as e:
            logger.error(f"Error segmenting Q&A region [{seg.start_u}-{seg.end_u}]: {e}")

    stored_boundaries = []
    for i, seg in enumerate(boundary_resp.segments):
        start_utt = _get_utterance_by_index(req.utterances, seg.start_u)
        end_utt = _get_utterance_by_index(req.utterances, seg.end_u)

        stored_boundaries.append(
            StoredBoundarySegment(
                seg_id=f"seg_{i:03d}",
                type=seg.type,
                start_u=seg.start_u,
                end_u=seg.end_u,
                start=start_utt.start if start_utt else 0.0,
                end=end_utt.end if end_utt else 0.0,
                confidence=seg.confidence,
                notes=seg.notes,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        )

    _build_and_save_export(req.video_id, stored_boundaries, all_blocks, req.utterances)

    return SegmentsResponse(
        video_id=req.video_id,
        boundary_segments=stored_boundaries,
        qa_blocks=all_blocks,
    )


def _extract_utterances_from_deepgram(deepgram_json: dict) -> list[Utterance]:
    """Extract utterances from raw Deepgram API response.

    Deepgram utterances have: start, end, transcript, id
    We convert to our format with: u (index), start, end, text
    """
    try:
        raw_utterances = deepgram_json.get("results", {}).get("utterances", [])
    except (KeyError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid Deepgram JSON: missing results.utterances",
        )

    if not raw_utterances:
        raise HTTPException(
            status_code=400,
            detail="No utterances found in Deepgram JSON. Ensure utterances=True in transcription.",
        )

    utterances = []
    for i, utt in enumerate(raw_utterances):
        utterances.append(
            Utterance(
                u=i,
                start=utt.get("start", 0.0),
                end=utt.get("end", 0.0),
                text=utt.get("transcript", ""),
            )
        )

    return utterances


@app.post("/segment/qa/from-deepgram", response_model=SegmentsResponse)
def segment_from_deepgram(req: DeepgramSegmentRequest) -> SegmentsResponse:
    """Segment from raw Deepgram JSON output.

    This is a convenience endpoint that accepts the full Deepgram transcription
    JSON file (as returned by src/transcribe.py), extracts utterances, and runs
    the full segmentation pipeline.

    Example:
        1. Transcribe video: uv run python src/transcribe.py "https://youtube.com/watch?v=VIDEO_ID"
        2. Send the resulting JSON to this endpoint

    The endpoint extracts utterances from results.utterances and runs Pass 1 + Pass 2.
    """
    utterances = _extract_utterances_from_deepgram(req.deepgram_json)

    full_req = FullSegmentRequest(video_id=req.video_id, utterances=utterances)
    return segment_full(full_req)


@app.post("/segment/qa/from-deepgram-file", response_model=SegmentsResponse)
def segment_from_deepgram_file(
    video_id: str = Query(..., description="Video identifier"),
    deepgram_json: dict = Body(..., description="Raw Deepgram API response JSON"),
) -> SegmentsResponse:
    """Segment from raw Deepgram JSON file upload.

    This endpoint is designed for easy file uploads via curl:

        curl -X POST "http://localhost:8001/segment/qa/from-deepgram-file?video_id=VIDEO_ID" \\
          -H "Content-Type: application/json" \\
          -d @data/transcripts/VIDEO_ID.json

    The video_id is passed as a query parameter, and the entire Deepgram JSON
    file is sent as the request body using curl's @file syntax.
    """
    utterances = _extract_utterances_from_deepgram(deepgram_json)

    full_req = FullSegmentRequest(video_id=video_id, utterances=utterances)
    return segment_full(full_req)


def _build_and_save_export(
    video_id: str,
    boundaries: list[StoredBoundarySegment],
    blocks: list[StoredQABlock],
    utterances: list[Utterance],
) -> None:
    """Build export JSON and save to file + database."""
    now = datetime.now(timezone.utc).isoformat()

    export_data = {
        "video_id": video_id,
        "boundary_segments": [
            {
                "seg_id": b.seg_id,
                "type": b.type,
                "start": b.start,
                "end": b.end,
                "start_u": b.start_u,
                "end_u": b.end_u,
                "confidence": b.confidence,
                "notes": b.notes,
            }
            for b in boundaries
        ],
        "qa_blocks": [
            {
                "block_id": b.block_id,
                "start": b.start,
                "end": b.end,
                "start_u": b.start_u,
                "end_u": b.end_u,
                "questions": b.questions,
                "answer_summary": b.answer_summary,
                "confidence": b.confidence,
                "text": b.text,
            }
            for b in blocks
        ],
        "created_at": now,
    }

    export_json = json.dumps(export_data, ensure_ascii=False, indent=2)
    upsert_export(video_id, export_json, now)

    export_path = EXPORTS_DIR / f"{video_id}.json"
    export_path.write_text(export_json, encoding="utf-8")
    logger.info(f"Export saved to {export_path}")


@app.get("/segments/{video_id}", response_model=SegmentsResponse)
def get_segments(video_id: str) -> SegmentsResponse:
    """Retrieve stored segmentation results for a video."""
    boundaries_data = get_boundaries(video_id)
    blocks_data = get_blocks(video_id)

    if not boundaries_data and not blocks_data:
        raise HTTPException(status_code=404, detail=f"No segments found for video: {video_id}")

    stored_boundaries = [
        StoredBoundarySegment(
            seg_id=b["seg_id"],
            type=b["type"],
            start_u=b["start_u"],
            end_u=b["end_u"],
            start=b["start"],
            end=b["end"],
            confidence=b["confidence"],
            notes=b["notes"],
            created_at=b["created_at"],
        )
        for b in boundaries_data
    ]

    stored_blocks = [
        StoredQABlock(
            block_id=b["block_id"],
            start_u=b["start_u"],
            end_u=b["end_u"],
            start=b["start"],
            end=b["end"],
            questions=json.loads(b["questions_json"]),
            answer_summary=b["answer_summary"],
            confidence=b["confidence"],
            text="",
            created_at=b["created_at"],
        )
        for b in blocks_data
    ]

    return SegmentsResponse(
        video_id=video_id,
        boundary_segments=stored_boundaries,
        qa_blocks=stored_blocks,
    )


@app.get("/exports/{video_id}", response_model=ExportResponse)
def get_video_export(video_id: str) -> ExportResponse:
    """Retrieve export JSON for a video."""
    export_data = get_export(video_id)

    if not export_data:
        raise HTTPException(status_code=404, detail=f"No export found for video: {video_id}")

    parsed = json.loads(export_data["export_json"])

    export_path = EXPORTS_DIR / f"{video_id}.json"
    file_path = str(export_path) if export_path.exists() else None

    return ExportResponse(
        video_id=video_id,
        export=ExportData(
            video_id=parsed["video_id"],
            boundary_segments=parsed["boundary_segments"],
            qa_blocks=parsed["qa_blocks"],
            created_at=parsed["created_at"],
        ),
        file_path=file_path,
    )


# =============================================================================
# HEALTH CHECK
# =============================================================================


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Health check endpoint for Docker orchestration."""
    return HealthResponse(
        status="healthy",
        model=OPENAI_MODEL,
        version=app.version,
    )
