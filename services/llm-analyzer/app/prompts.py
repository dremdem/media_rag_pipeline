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
    where each block represents one answered question (or bundle of questions).
    """
    # Format utterances compactly
    utterance_lines = []
    for u in utterances:
        text_preview = u.text[:100] + "..." if len(u.text) > 100 else u.text
        utterance_lines.append(f'[u={u.u} {u.start:.1f}-{u.end:.1f}] "{text_preview}"')

    utterances_text = "\n".join(utterance_lines)

    payload = {
        "task": "Segment this Russian Q&A transcript into semantic answer blocks.",
        "language": "ru",
        "context": (
            "This is the Q&A portion of a Russian political commentary video. "
            "The host reads viewer questions/comments and responds to them. "
            "Each block should represent one question-answer exchange."
        ),
        "definitions": {
            "qa_block": (
                "A continuous segment where the speaker answers one question OR "
                "a bundle of related questions. Each block should be semantically coherent - "
                "it addresses one topic/question before moving to the next."
            ),
            "question": (
                "The viewer's question being answered. May be read aloud by the speaker, "
                "paraphrased, or implied from context. Extract if detectable."
            ),
        },
        "block_boundary_markers_ru": {
            "description": "Signs that a new Q&A block is starting:",
            "markers": [
                "New viewer name mentioned (e.g., 'Виктор пишет...', 'Анна спрашивает...')",
                "Explicit topic change (e.g., 'Теперь о...', 'Следующий вопрос...')",
                "New question being read (e.g., 'Вопрос от...', 'Ещё один вопрос...')",
            ],
        },
        "input": {
            "qa_range": qa_range,
            "total_utterances": len(utterances),
            "utterances": utterances_text,
        },
        "output_schema": {
            "qa_blocks": [
                {
                    "start_u": "integer (utterance index)",
                    "end_u": "integer (utterance index, inclusive)",
                    "questions": ["array of question strings if detectable, empty if not"],
                    "answer_summary": "1-2 sentence summary of the answer",
                    "confidence": "number 0..1",
                }
            ]
        },
        "rules": [
            "Return ONLY valid JSON. No extra text.",
            "start_u and end_u MUST be within the provided qa_range.",
            "Blocks MUST cover the entire Q&A range without gaps or overlaps.",
            "Blocks MUST be ordered by start_u.",
            "Look for viewer names - each new name often signals a new block.",
            "questions array may be empty if the question is not detectable from transcript.",
            "answer_summary should be 1-2 sentences describing the main point of the answer.",
            "Do NOT generate fake question text. Only extract if clearly present.",
            "Do NOT copy long text into answer_summary. Keep it brief.",
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
