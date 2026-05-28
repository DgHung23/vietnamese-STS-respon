import argparse
import os
import sys
from typing import Iterable, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from response_engine import HybridRAGResponder, LLMError, RAGError, config_from_env
from speech_to_text import add_stt_arguments, list_input_devices, settings_from_args
from speech_to_text import stream_realtime_transcripts
from text_to_speech import FPTTTSClient, TTSError


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def build_arg_parser() -> argparse.ArgumentParser:
    if load_dotenv is not None:
        load_dotenv()

    parser = argparse.ArgumentParser(
        description="STT -> RAG/LLM -> TTS pipeline for Greenwich Vietnam FAQ."
    )
    add_stt_arguments(parser)

    parser.add_argument(
        "--text",
        help="Process this text directly instead of listening to the microphone.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one final transcript and exit.",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        default=_env_bool("NO_TTS", False),
        help="Print the answer without reading it aloud.",
    )
    parser.add_argument(
        "--faq-path",
        default=os.getenv("FAQ_PATH", "faq.csv"),
        help="Path to the FAQ CSV file. Default: faq.csv",
    )
    parser.add_argument(
        "--rag-threshold",
        type=float,
        default=_env_float("RAG_THRESHOLD", 0.72),
        help="Minimum similarity score required to use an FAQ answer.",
    )
    parser.add_argument(
        "--rag-debug",
        action="store_true",
        default=_env_bool("RAG_DEBUG", False),
        help="Print the selected FAQ match and routing decision.",
    )
    parser.add_argument(
        "--gemini-model",
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model used when the FAQ has no confident match.",
    )
    parser.add_argument(
        "--tts-voice",
        default=os.getenv("FPT_TTS_VOICE", "banmai"),
        help="FPT.AI TTS voice. Default: banmai",
    )
    parser.add_argument(
        "--tts-speed",
        default=os.getenv("FPT_TTS_SPEED", "1"),
        help="FPT.AI TTS speed. Default: 1",
    )
    return parser


def build_responder(args: argparse.Namespace) -> HybridRAGResponder:
    config = config_from_env(
        faq_path=args.faq_path,
        threshold=args.rag_threshold,
        gemini_model=args.gemini_model,
        debug=args.rag_debug,
        verbose=True,
    )
    return HybridRAGResponder(config)


def build_tts_client(args: argparse.Namespace) -> Optional[FPTTTSClient]:
    if args.no_tts:
        return None

    try:
        return FPTTTSClient.from_env(voice=args.tts_voice, speed=args.tts_speed)
    except TTSError as exc:
        print(f"[TTS] Disabled: {exc}", file=sys.stderr, flush=True)
        return None


def iter_transcripts(args: argparse.Namespace) -> Iterable[str]:
    if args.text:
        yield args.text
        return

    settings = settings_from_args(args)
    print("Listening for Vietnamese speech. Press Ctrl+C to stop.")
    print(f"Language: {settings.language_code}, model: {settings.model}")
    yield from stream_realtime_transcripts(settings)


def process_query(
    transcript: str,
    responder: HybridRAGResponder,
    tts_client: Optional[FPTTTSClient],
) -> None:
    print(f"\nSTT: {transcript}", flush=True)

    try:
        answer = responder.get_response(transcript)
    except (RAGError, LLMError) as exc:
        print(f"LLM/RAG error: {exc}", flush=True)
        return

    print(f"AI: {answer}", flush=True)

    if tts_client is None:
        return

    try:
        tts_client.speak(answer)
    except TTSError as exc:
        print(f"TTS error: {exc}", flush=True)


def main() -> None:
    configure_stdio()
    args = build_arg_parser().parse_args()

    if args.list_devices:
        list_input_devices()
        return

    responder = build_responder(args)
    tts_client = build_tts_client(args)

    try:
        for transcript in iter_transcripts(args):
            if not transcript.strip():
                continue

            process_query(transcript, responder, tts_client)
            if args.once or args.text:
                break
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
