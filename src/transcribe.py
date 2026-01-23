#!/usr/bin/env python3
"""Transcribe YouTube videos or local audio files using Deepgram API.

This script downloads audio from a YouTube video (or uses a local file),
transcribes it using Deepgram's API, and saves the results as JSON, SRT,
and plain text files.

Usage:
    # From YouTube URL
    uv run python src/transcribe.py "https://youtube.com/watch?v=VIDEO_ID"
    uv run python src/transcribe.py "https://youtube.com/watch?v=VIDEO_ID" --language en

    # From local audio file
    uv run python src/transcribe.py --audio-file path/to/audio.mp3
    uv run python src/transcribe.py -a path/to/audio.mp3 --language en

Environment Variables:
    DEEPGRAM_API_KEY: Required. Your Deepgram API key.

Requirements:
    - ffmpeg must be installed on the system for audio extraction
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import httpx
import yt_dlp
from deepgram import DeepgramClient
from deepgram_captions import DeepgramConverter, srt
from dotenv import load_dotenv

load_dotenv()

# Configuration
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
DEFAULT_OUTPUT_DIR = "data/transcripts"
DEFAULT_LANGUAGE = "ru"


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$",  # Just the ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_audio(url: str, output_dir: Path) -> Path:
    """Download audio from YouTube video.

    Args:
        url: YouTube video URL
        output_dir: Directory to save the audio file

    Returns:
        Path to the downloaded audio file
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from URL: {url}")

    output_template = str(output_dir / f"{video_id}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": False,
        "no_warnings": False,
    }

    print(f"Downloading audio from: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    audio_path = output_dir / f"{video_id}.mp3"
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not created: {audio_path}")

    print(f"Audio saved: {audio_path}")
    return audio_path


def transcribe_audio(
    audio_path: Path,
    language: str = DEFAULT_LANGUAGE,
    diarize: bool = False,
    filler_words: bool = False,
    timeout: int = 600,
) -> dict:
    """Transcribe audio file using Deepgram API.

    Args:
        audio_path: Path to the audio file
        language: Language code (e.g., 'ru', 'en')
        diarize: Enable speaker diarization
        filler_words: Include filler words like 'um', 'uh'
        timeout: API timeout in seconds (default: 600 for large files)

    Returns:
        Deepgram API response as dictionary
    """
    if not DEEPGRAM_API_KEY:
        raise RuntimeError(
            "DEEPGRAM_API_KEY is not set. "
            "Please set it in your environment or .env file."
        )

    # Get file size for info
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)

    print(f"Transcribing: {audio_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Language: {language}")
    print(f"  Diarize: {diarize}")
    print(f"  Filler words: {filler_words}")
    print(f"  Timeout: {timeout}s")

    # Create client with custom httpx client for longer timeout
    http_client = httpx.Client(timeout=httpx.Timeout(timeout, connect=30.0))
    client = DeepgramClient(api_key=DEEPGRAM_API_KEY, httpx_client=http_client)

    with open(audio_path, "rb") as audio_file:
        audio_data = audio_file.read()

    # Detect mimetype based on file extension
    suffix = audio_path.suffix.lower()
    mimetype_map = {
        ".mp3": "audio/mp3",
        ".wav": "audio/wav",
        ".m4a": "audio/m4a",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }
    mimetype = mimetype_map.get(suffix, "audio/mp3")

    # Transcribe with options using v1 API
    response = client.listen.v1.media.transcribe_file(
        request=audio_data,
        model="nova-3",
        language=language,
        smart_format=True,
        punctuate=True,
        paragraphs=True,
        utterances=True,  # Required for SRT generation
        filler_words=filler_words,
        diarize=diarize,
    )

    # Convert response to dict - handle different SDK versions
    if hasattr(response, "to_dict"):
        return response.to_dict()
    elif hasattr(response, "to_json"):
        return json.loads(response.to_json())
    elif hasattr(response, "model_dump"):
        return response.model_dump()
    elif hasattr(response, "result"):
        # v1 API might wrap result
        result = response.result
        if hasattr(result, "to_dict"):
            return result.to_dict()
        elif hasattr(result, "to_json"):
            return json.loads(result.to_json())
    # Fallback: try to access as dict-like or use __dict__
    try:
        return dict(response)
    except (TypeError, ValueError):
        return response.__dict__


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt_from_paragraphs(response: dict) -> str:
    """Generate SRT content from paragraphs instead of utterances.

    This produces longer, more readable subtitle segments compared to
    the default utterance-based generation.

    Args:
        response: Deepgram API response dictionary

    Returns:
        SRT formatted string
    """
    paragraphs = (
        response.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [{}])[0]
        .get("paragraphs", {})
        .get("paragraphs", [])
    )

    if not paragraphs:
        raise ValueError("No paragraphs found in response. Ensure paragraphs=True in API call.")

    srt_lines = []
    for i, para in enumerate(paragraphs, 1):
        start = format_srt_timestamp(para["start"])
        end = format_srt_timestamp(para["end"])
        # Combine all sentences in the paragraph
        text = " ".join(s["text"] for s in para.get("sentences", []))
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    return "\n".join(srt_lines)


def save_results(
    response: dict,
    output_dir: Path,
    video_id: str,
    srt_mode: str = "utterances",
) -> tuple[Path, Path, Path]:
    """Save transcription results to files.

    Args:
        response: Deepgram API response
        output_dir: Output directory
        video_id: YouTube video ID
        srt_mode: SRT generation mode - "utterances" (short segments) or
                  "paragraphs" (longer, more readable segments)

    Returns:
        Tuple of (json_path, srt_path, txt_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Custom JSON encoder for datetime objects
    def json_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    # Save full JSON response
    json_path = output_dir / f"{video_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2, default=json_serializer)
    print(f"JSON saved: {json_path}")

    # Extract plain text transcript
    transcript = ""
    try:
        channels = response.get("results", {}).get("channels", [])
        if channels:
            alternatives = channels[0].get("alternatives", [])
            if alternatives:
                transcript = alternatives[0].get("transcript", "")
    except (KeyError, IndexError) as e:
        print(f"Warning: Could not extract transcript: {e}")

    # Save plain text
    txt_path = output_dir / f"{video_id}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"TXT saved: {txt_path}")

    # Generate and save SRT
    srt_path = output_dir / f"{video_id}.srt"
    try:
        if srt_mode == "paragraphs":
            srt_content = generate_srt_from_paragraphs(response)
        else:
            # Default: use deepgram-captions with utterances
            converter = DeepgramConverter(response)
            srt_content = srt(converter)
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        print(f"SRT saved: {srt_path} (mode: {srt_mode})")
    except Exception as e:
        print(f"Warning: Could not generate SRT: {e}")
        srt_path = None

    return json_path, srt_path, txt_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Transcribe YouTube videos or local audio files using Deepgram API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From YouTube URL
  %(prog)s "https://youtube.com/watch?v=VIDEO_ID"
  %(prog)s "https://youtu.be/VIDEO_ID" --language en

  # From local audio file
  %(prog)s --audio-file path/to/audio.mp3
  %(prog)s -a recording.wav --language en --diarize

Environment:
  DEEPGRAM_API_KEY    Your Deepgram API key (required)

Note:
  ffmpeg must be installed for audio extraction from YouTube.
  Install with: brew install ffmpeg (macOS)
        """,
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="YouTube video URL (optional if --audio-file is provided)",
    )
    parser.add_argument(
        "-a",
        "--audio-file",
        type=Path,
        help="Path to local audio file (skip YouTube download)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language code (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization",
    )
    parser.add_argument(
        "--filler-words",
        action="store_true",
        help="Include filler words like 'um', 'uh' (default: remove them)",
    )
    parser.add_argument(
        "--delete-audio",
        action="store_true",
        help="Delete audio file after transcription (default: keep)",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=600,
        help="API timeout in seconds (default: 600 for large files)",
    )
    parser.add_argument(
        "--srt-mode",
        choices=["utterances", "paragraphs"],
        default="utterances",
        help="SRT generation mode: 'utterances' for short segments (default), "
        "'paragraphs' for longer, more readable segments",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Validate API key
    if not DEEPGRAM_API_KEY:
        print("Error: DEEPGRAM_API_KEY is not set.", file=sys.stderr)
        print("Please set it in your environment or .env file.", file=sys.stderr)
        return 1

    # Validate inputs: need either URL or audio file
    if not args.url and not args.audio_file:
        print("Error: Must provide either a YouTube URL or --audio-file.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Determine source: local file or YouTube
        if args.audio_file:
            # Use local audio file
            audio_path = args.audio_file
            if not audio_path.exists():
                print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
                return 1
            # Use filename stem as the ID
            file_id = audio_path.stem
            print(f"Using local audio file: {audio_path}")
            downloaded = False
        else:
            # Download from YouTube
            video_id = extract_video_id(args.url)
            if not video_id:
                print(f"Error: Could not extract video ID from: {args.url}", file=sys.stderr)
                return 1
            audio_path = download_audio(args.url, output_dir)
            file_id = video_id
            downloaded = True

        # Transcribe
        response = transcribe_audio(
            audio_path,
            language=args.language,
            diarize=args.diarize,
            filler_words=args.filler_words,
            timeout=args.timeout,
        )

        # Save results
        json_path, srt_path, txt_path = save_results(
            response, output_dir, file_id, srt_mode=args.srt_mode
        )

        # Optionally delete audio (only if downloaded)
        if args.delete_audio and downloaded:
            audio_path.unlink()
            print(f"Audio deleted: {audio_path}")

        print("\nTranscription complete!")
        print(f"  ID: {file_id}")
        print(f"  JSON: {json_path}")
        print(f"  SRT: {srt_path}")
        print(f"  TXT: {txt_path}")
        if not args.delete_audio or not downloaded:
            print(f"  Audio: {audio_path}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
