"""Prompt templates for LLM Analyzer Service.

This module contains prompt templates for:
1. Opinion detection
2. Q&A boundary segmentation (Pass 1)
3. Q&A block segmentation (Pass 2)

## Prompt Design Notes

### Russian Political Commentary Videos

The prompts are tuned for Russian-language political commentary/news videos
where the host typically:
1. Starts with news/analysis (narrative section)
2. Transitions to answering viewer questions (Q&A section)

### Key Transition Markers (Russian)

Explicit transitions to Q&A:
- "ответы на ваши вопросы" (answers to your questions)
- "перейдём к вопросам" (let's move to questions)
- "теперь вопросы" (now questions)

Q&A indicators:
- "[Имя], к вопросу о..." (addressing viewer by name)
- "[Имя] пишет/спрашивает..." (viewer writes/asks)
- "вопрос от [Имя]..." (question from viewer)

### Why This Matters

Without these markers, the LLM may:
- Mark entire transcript as "narrative" (missing Q&A section)
- Only detect Q&A at the very end (missing early transitions)
- Miss transitions in the middle of long transcripts

### Q&A Block Segmentation (Pass 2)

Key principle: ONE block = ONE viewer's question + host's COMPLETE answer.

Block boundary detection:
- NEW block starts ONLY when a NEW VIEWER NAME appears
- Pattern: "[Имя]. ..." or "[Имя], ..." or "[Имя] пишет..."
- Examples: "Виктор.", "Ольга,", "Андрей пишет:", "Поле из Сум."

What is NOT a new block:
- Topic change within the same answer
- Host continuing reasoning (utterances with "но", "и", "также")
- Host addressing viewer mid-answer ("уважаемая/уважаемый")

Question extraction rules:
- NEVER hallucinate questions
- Only extract if LITERALLY quoted in transcript
- If question not read aloud, use empty array []

Expected blocks: ~15-30 for 1 hour of Q&A (not 80+!)

### Model Selection

- Pass 1 (boundaries): Uses OPENAI_MODEL (default: gpt-4o-mini)
- Pass 2 (blocks): Uses OPENAI_MODEL_BLOCKS (default: gpt-4o)

Block segmentation uses a stronger model because:
- Large context (~50k tokens for long Q&A sections)
- Complex viewer name detection rules
- Requires better instruction following
"""

import json
from typing import Any

from .schemas import DetectRequest, Utterance


def build_opinion_prompt(req: DetectRequest) -> str:
    """Build prompt for opinion detection."""
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


def build_boundary_prompt(utterances: list[Utterance]) -> str:
    """Build prompt for Pass 1: boundary detection (Narrative vs Q&A).

    The prompt instructs the LLM to identify where narrative content ends
    and Q&A content begins, returning utterance index ranges.

    Key improvement: Includes Russian-specific transition markers and examples
    to help detect Q&A sections in political commentary videos.
    """
    # Format utterances compactly for the prompt
    utterance_lines = []
    for u in utterances:
        # Format: [u=0 0.0-3.2] "text..."
        text_preview = u.text[:100] + "..." if len(u.text) > 100 else u.text
        utterance_lines.append(f'[u={u.u} {u.start:.1f}-{u.end:.1f}] "{text_preview}"')

    utterances_text = "\n".join(utterance_lines)

    payload = {
        "task": "Segment this Russian transcript into NARRATIVE and Q&A regions.",
        "language": "ru",
        "context": (
            "This is a Russian political commentary/news video. "
            "The host typically starts with news/analysis (narrative), "
            "then transitions to answering viewer questions (Q&A). "
            "Look carefully for transition phrases in the MIDDLE of the transcript."
        ),
        "definitions": {
            "narrative": (
                "Monologue content: news, analysis, commentary, storytelling. "
                "The speaker is delivering information without responding to audience questions."
            ),
            "qa": (
                "Question-and-answer content: the speaker is answering viewer questions, "
                "reading comments, or addressing audience by name. "
                "The speaker responds to specific people or their questions."
            ),
        },
        "transition_markers_ru": {
            "description": "Common Russian phrases that signal transition to Q&A section:",
            "explicit_transitions": [
                "ответы на ваши вопросы",
                "перейдём к вопросам",
                "отвечу на вопросы",
                "ваши вопросы и комментарии",
                "перейти к ответам",
                "теперь вопросы",
            ],
            "qa_indicators": [
                "[Имя], к вопросу о... (addressing viewer by name)",
                "[Имя] пишет/спрашивает... (viewer writes/asks)",
                "вопрос от [Имя]... (question from viewer)",
                "комментарий от [Имя]... (comment from viewer)",
                "читаю комментарий... (reading a comment)",
            ],
            "note": "When you see these patterns, Q&A section has likely begun!",
        },
        "input": {
            "total_utterances": len(utterances),
            "first_index": utterances[0].u if utterances else 0,
            "last_index": utterances[-1].u if utterances else 0,
            "utterances": utterances_text,
        },
        "output_schema": {
            "segments": [
                {
                    "type": "narrative OR qa",
                    "start_u": "integer (utterance index)",
                    "end_u": "integer (utterance index, inclusive)",
                    "confidence": "number 0..1",
                    "notes": "short description of segment content (optional)",
                }
            ]
        },
        "rules": [
            "Return ONLY valid JSON. No extra text.",
            "IMPORTANT: Scan the ENTIRE transcript for transition markers, not just beginning/end!",
            "start_u and end_u MUST be valid utterance indices from the input.",
            "Segments MUST cover the entire range without gaps or overlaps.",
            "Segments MUST be ordered by start_u.",
            "If the entire transcript is narrative, return one segment with type='narrative'.",
            "If the entire transcript is Q&A, return one segment with type='qa'.",
            "If you find a transition marker, the Q&A section starts AT or BEFORE that utterance.",
            "notes should be 1-2 sentences max, describing what happens in that segment.",
            "Do NOT invent text. Only use indices and short notes.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def build_blocks_prompt(utterances: list[Utterance], qa_range: dict[str, int]) -> str:
    """Build prompt for Pass 2: Q&A block segmentation.

    Given only the Q&A region utterances, split into semantic blocks
    where each block represents one viewer's question + the host's complete answer.

    Key improvements:
    - Detect viewer names as block boundaries
    - Prevent over-segmentation (expect 15-30 blocks for 1h Q&A)
    - Never hallucinate questions - only extract if literally quoted
    - Include full answer in one block, even if spans many utterances
    """
    # Format utterances compactly
    utterance_lines = []
    for u in utterances:
        text_preview = u.text[:100] + "..." if len(u.text) > 100 else u.text
        utterance_lines.append(f'[u={u.u} {u.start:.1f}-{u.end:.1f}] "{text_preview}"')

    utterances_text = "\n".join(utterance_lines)

    # Estimate expected block count (roughly 1 block per 2-4 minutes of Q&A)
    total_duration = utterances[-1].end - utterances[0].start if utterances else 0
    estimated_blocks = max(5, min(40, int(total_duration / 150)))  # ~2.5 min per block

    payload = {
        "task": "Segment this Russian Q&A transcript into semantic answer blocks.",
        "language": "ru",
        "context": (
            "This is the Q&A portion of a Russian political commentary video. "
            "The host reads viewer questions/comments and gives COMPLETE answers. "
            "One block = one viewer's question + the host's FULL answer (may span many utterances). "
            "Do NOT split the host's answer into multiple blocks!"
        ),
        "critical_rules": {
            "what_is_a_block": (
                "ONE block = ONE viewer's question/comment + the host's COMPLETE answer. "
                "The answer may be long (5-20 utterances) - that's still ONE block. "
                "A new block starts ONLY when a NEW VIEWER is addressed."
            ),
            "how_to_detect_new_block": (
                "A NEW block starts when you see a NEW VIEWER NAME at the START of an utterance. "
                "Pattern: '[Имя]. ...' or '[Имя], ...' or '[Имя] пишет/спрашивает...' "
                "Examples: 'Виктор.', 'Ольга,', 'Андрей пишет:', 'Поле из Сум.' "
                "If no new viewer name → it's the SAME block (continuation of answer)."
            ),
            "what_is_NOT_a_new_block": [
                "Topic change within the same answer",
                "Host saying 'уважаемая/уважаемый' mid-answer",
                "Utterances starting with 'но', 'и', 'также', 'потому что'",
                "Any continuation of the host's reasoning",
            ],
        },
        "question_extraction_rules": {
            "NEVER_HALLUCINATE": (
                "CRITICAL: The 'questions' field must contain ONLY text that is "
                "LITERALLY QUOTED in the transcript. If the viewer's question is not "
                "read aloud by the host, leave questions array EMPTY []."
            ),
            "correct_examples": [
                "Text: 'Виктор спрашивает: не Волков ли подтолкнул Навального?' → questions: ['не Волков ли подтолкнул Навального?']",
                "Text: 'Ольга. Как вы относитесь к этому?' → questions: ['Как вы относитесь к этому?']",
            ],
            "incorrect_examples": [
                "WRONG: Text about Lithuania → questions: ['околосмертные переживания'] (hallucinated!)",
                "WRONG: Inventing a question that sounds related but isn't in the text",
            ],
            "when_to_leave_empty": (
                "If the host just starts answering without reading the question, "
                "or if the question is not clearly stated, use questions: []"
            ),
        },
        "expected_output": {
            "block_count": f"For this transcript, expect approximately {estimated_blocks} blocks (±30%)",
            "block_size": "Each block typically spans 5-30 utterances (the host's full answer)",
            "warning": "If you have >50 blocks, you are over-segmenting! Merge blocks.",
        },
        "example_correct_segmentation": {
            "description": "How blocks should look:",
            "blocks": [
                {
                    "comment": "Block starts with viewer name 'Виктор'",
                    "start_text": "Виктор, к вопросу о ФБК...",
                    "spans": "15 utterances (host's full answer about FBK)",
                    "questions": ["к вопросу о ФБК"],
                },
                {
                    "comment": "NEW block because NEW viewer 'Ольга' appears",
                    "start_text": "Ольга. Как вы относитесь к свободе слова?",
                    "spans": "20 utterances (host's full answer about freedom of speech)",
                    "questions": ["Как вы относитесь к свободе слова?"],
                },
                {
                    "comment": "NEW block because NEW viewer 'Андрей' appears",
                    "start_text": "Андрей пишет: поймал себя на мысли...",
                    "spans": "10 utterances (host's response)",
                    "questions": [],  # Question not clearly stated
                },
            ],
        },
        "example_incorrect_segmentation": {
            "description": "What NOT to do:",
            "mistakes": [
                "WRONG: Splitting host's answer about Lithuania into 3 blocks",
                "WRONG: Creating new block because host changed sub-topic",
                "WRONG: Creating new block for every few utterances",
                "WRONG: Inventing questions that aren't in the transcript",
            ],
        },
        "input": {
            "qa_range": qa_range,
            "total_utterances": len(utterances),
            "estimated_blocks": estimated_blocks,
            "utterances": utterances_text,
        },
        "output_schema": {
            "qa_blocks": [
                {
                    "start_u": "integer (utterance index where viewer name appears)",
                    "end_u": "integer (last utterance of host's answer, inclusive)",
                    "questions": "array of LITERAL quotes from transcript, or [] if not stated",
                    "answer_summary": "1-2 sentence summary of what the host said",
                    "confidence": "number 0..1",
                }
            ]
        },
        "rules": [
            "Return ONLY valid JSON. No extra text.",
            "start_u and end_u MUST be within the provided qa_range.",
            "Blocks MUST cover the entire Q&A range without gaps or overlaps.",
            "Blocks MUST be ordered by start_u.",
            "NEW BLOCK = NEW VIEWER NAME at start of utterance. Nothing else!",
            "One viewer's question + host's FULL answer = ONE block (even if 20+ utterances).",
            "questions array: ONLY literal quotes from transcript, or empty [].",
            "NEVER invent or hallucinate questions. If unsure, use [].",
            "answer_summary: Brief description of the host's response (1-2 sentences).",
            f"Expected block count: ~{estimated_blocks}. If you have >50 blocks, merge them!",
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


# System prompts for different tasks

SYSTEM_PROMPT_EXTRACTION = (
    "You are a strict information extraction engine. "
    "Return ONLY valid JSON matching the requested schema. "
    "Never add explanations or commentary outside the JSON."
)

SYSTEM_PROMPT_SEGMENTATION = (
    "You are a transcript segmentation engine. "
    "Your task is to identify structural boundaries in speech transcripts. "
    "Return ONLY valid JSON with utterance indices. Never invent or copy text content."
)
