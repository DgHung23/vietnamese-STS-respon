import argparse
import os
import queue
import sys
import time
from dataclasses import dataclass
from typing import Generator, Iterable, Optional

import pyaudio
from google.api_core import exceptions as google_exceptions
from google.cloud import speech

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)
STREAMING_LIMIT_SECONDS = 240


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class Settings:
    language_code: str
    model: str
    sample_rate: int
    chunk_size: int
    device_index: Optional[int]
    streaming_limit_seconds: int
    interim_results: bool


class MicrophoneStream:
    """Opens a microphone stream and yields raw 16-bit PCM audio chunks."""

    def __init__(
        self,
        rate: int,
        chunk_size: int,
        device_index: Optional[int] = None,
    ) -> None:
        self.rate = rate
        self.chunk_size = chunk_size
        self.device_index = device_index
        self.buffer: queue.Queue[Optional[bytes]] = queue.Queue()
        self.closed = True
        self.audio_interface: Optional[pyaudio.PyAudio] = None
        self.audio_stream: Optional[pyaudio.Stream] = None

    def __enter__(self) -> "MicrophoneStream":
        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.closed = True
        self.buffer.put(None)

        if self.audio_stream is not None:
            self.audio_stream.stop_stream()
            self.audio_stream.close()

        if self.audio_interface is not None:
            self.audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self.buffer.put(in_data)
        return None, pyaudio.paContinue

    def generator(self, max_seconds: int) -> Generator[bytes, None, None]:
        deadline = time.monotonic() + max_seconds

        while not self.closed and time.monotonic() < deadline:
            chunk = self.buffer.get()
            if chunk is None:
                return

            data = [chunk]
            while True:
                try:
                    chunk = self.buffer.get(block=False)
                except queue.Empty:
                    break

                if chunk is None:
                    return
                data.append(chunk)

            yield b"".join(data)


def list_input_devices() -> None:
    audio = pyaudio.PyAudio()
    try:
        print("Input devices:")
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) > 0:
                print(f"  {index}: {info.get('name')}")
    finally:
        audio.terminate()


def build_streaming_config(settings: Settings) -> speech.StreamingRecognitionConfig:
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=settings.sample_rate,
        language_code=settings.language_code,
        model=settings.model,
        enable_automatic_punctuation=True,
        audio_channel_count=1,
    )
    return speech.StreamingRecognitionConfig(
        config=config,
        interim_results=settings.interim_results,
        single_utterance=False,
    )


def request_stream(
    audio_chunks: Iterable[bytes],
) -> Generator[speech.StreamingRecognizeRequest, None, None]:
    for chunk in audio_chunks:
        yield speech.StreamingRecognizeRequest(audio_content=chunk)


def final_transcripts(
    responses: Iterable[speech.StreamingRecognizeResponse],
) -> Generator[str, None, None]:
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives or not result.is_final:
            continue

        transcript = result.alternatives[0].transcript.strip()
        if transcript:
            yield transcript


def print_responses(responses: Iterable[speech.StreamingRecognizeResponse]) -> None:
    previous_interim_length = 0

    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript.strip()
        if not transcript:
            continue

        if result.is_final:
            sys.stdout.write("\r" + " " * previous_interim_length + "\r")
            print(transcript, flush=True)
            previous_interim_length = 0
            continue

        overwrite = " " * max(previous_interim_length - len(transcript), 0)
        sys.stdout.write(f"\r{transcript}{overwrite}")
        sys.stdout.flush()
        previous_interim_length = len(transcript)


def stream_realtime_transcripts(settings: Settings) -> Generator[str, None, None]:
    client = speech.SpeechClient()
    streaming_config = build_streaming_config(settings)

    with MicrophoneStream(
        rate=settings.sample_rate,
        chunk_size=settings.chunk_size,
        device_index=settings.device_index,
    ) as microphone:
        while not microphone.closed:
            audio_generator = microphone.generator(settings.streaming_limit_seconds)
            requests = request_stream(audio_generator)

            try:
                responses = client.streaming_recognize(streaming_config, requests)
                yield from final_transcripts(responses)
            except google_exceptions.OutOfRange:
                continue
            except google_exceptions.GoogleAPICallError as exc:
                print(f"\nGoogle Speech-to-Text error: {exc}", file=sys.stderr)
                raise


def run_realtime_transcription(settings: Settings) -> None:
    print("Listening for Vietnamese speech. Press Ctrl+C to stop.")
    print(f"Language: {settings.language_code}, model: {settings.model}")

    for transcript in stream_realtime_transcripts(settings):
        print(transcript, flush=True)


def add_stt_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--language-code",
        default=os.getenv("STT_LANGUAGE_CODE", "vi-VN"),
        help="BCP-47 language code. Default: vi-VN",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("STT_MODEL", "latest_long"),
        help="Google Speech-to-Text V1 model. Default: latest_long",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=int(os.getenv("STT_SAMPLE_RATE", SAMPLE_RATE)),
        help=f"Microphone sample rate in Hz. Default: {SAMPLE_RATE}",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=(
            int(os.environ["STT_DEVICE_INDEX"])
            if os.getenv("STT_DEVICE_INDEX")
            else None
        ),
        help="PyAudio input device index. Use --list-devices to find one.",
    )
    parser.add_argument(
        "--streaming-limit",
        type=int,
        default=int(os.getenv("STT_STREAMING_LIMIT", STREAMING_LIMIT_SECONDS)),
        help=f"Seconds per API stream before reconnecting. Default: {STREAMING_LIMIT_SECONDS}",
    )
    parser.add_argument(
        "--no-interim",
        action="store_true",
        help="Only print finalized transcripts.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List microphone input devices and exit.",
    )
    return parser


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime Vietnamese speech-to-text using Google Cloud Speech-to-Text."
    )
    return add_stt_arguments(parser)


def parse_args() -> argparse.Namespace:
    if load_dotenv is not None:
        load_dotenv()

    parser = build_arg_parser()
    return parser.parse_args()


def settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings(
        language_code=args.language_code,
        model=args.model,
        sample_rate=args.sample_rate,
        chunk_size=int(args.sample_rate / 10),
        device_index=args.device_index,
        streaming_limit_seconds=args.streaming_limit,
        interim_results=not args.no_interim,
    )


def main() -> None:
    configure_stdio()
    args = parse_args()

    if args.list_devices:
        list_input_devices()
        return

    settings = settings_from_args(args)

    try:
        run_realtime_transcription(settings)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
