import io
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


FPT_TTS_URL = "https://api.fpt.ai/hmi/tts/v5"
DEFAULT_TTS_VOICE = "banmai"
DEFAULT_TTS_SPEED = "1"
DEFAULT_TTS_POLL_DELAY = 2.0
DEFAULT_TTS_TIMEOUT = 30
DEFAULT_TTS_MAX_WAIT = 30.0
RETRYABLE_DOWNLOAD_STATUSES = {404, 408, 425, 429, 500, 502, 503, 504}


class TTSError(RuntimeError):
    """Raised when text-to-speech cannot synthesize or play audio."""


@dataclass
class FPTTTSConfig:
    api_key: str
    voice: str = DEFAULT_TTS_VOICE
    speed: str = DEFAULT_TTS_SPEED
    api_url: str = FPT_TTS_URL
    poll_delay: float = DEFAULT_TTS_POLL_DELAY
    timeout: int = DEFAULT_TTS_TIMEOUT
    max_wait: float = DEFAULT_TTS_MAX_WAIT


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def add_ffmpeg_to_path() -> None:
    ffmpeg_dir = os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Microsoft",
        "WinGet",
        "Links",
    )

    if os.path.exists(os.path.join(ffmpeg_dir, "ffmpeg.exe")):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


def config_from_env(**overrides) -> FPTTTSConfig:
    if load_dotenv is not None:
        load_dotenv()

    api_key = (
        overrides.pop("api_key", None)
        or os.getenv("FPT_API_KEY")
        or os.getenv("FPT_TTS_API_KEY")
    )
    if not api_key:
        raise TTSError("Missing FPT TTS API key. Set FPT_API_KEY in .env.")

    config = FPTTTSConfig(
        api_key=api_key,
        voice=os.getenv("FPT_TTS_VOICE", DEFAULT_TTS_VOICE),
        speed=os.getenv("FPT_TTS_SPEED", DEFAULT_TTS_SPEED),
        api_url=os.getenv("FPT_TTS_URL", FPT_TTS_URL),
        poll_delay=_env_float("FPT_TTS_POLL_DELAY", DEFAULT_TTS_POLL_DELAY),
        timeout=_env_int("FPT_TTS_TIMEOUT", DEFAULT_TTS_TIMEOUT),
        max_wait=_env_float("FPT_TTS_MAX_WAIT", DEFAULT_TTS_MAX_WAIT),
    )

    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    return config


class FPTTTSClient:
    """Small FPT.AI TTS client used by the voice pipeline."""

    def __init__(self, config: FPTTTSConfig) -> None:
        self.config = config
        add_ffmpeg_to_path()

    @classmethod
    def from_env(cls, **overrides) -> "FPTTTSClient":
        return cls(config_from_env(**overrides))

    def create_audio_url(self, text: str) -> str:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise TTSError("Cannot synthesize empty text.")

        headers = {
            "api-key": self.config.api_key,
            "speed": self.config.speed,
            "voice": self.config.voice,
        }
        response = requests.post(
            self.config.api_url,
            data=cleaned_text.encode("utf-8"),
            headers=headers,
            timeout=self.config.timeout,
        )

        if response.status_code != 200:
            raise TTSError(f"FPT.AI TTS API returned HTTP {response.status_code}.")

        try:
            result = response.json()
        except ValueError as exc:
            raise TTSError("FPT.AI TTS API returned invalid JSON.") from exc

        if result.get("error") != 0:
            raise TTSError(f"FPT.AI TTS error: {result.get('message')}")

        audio_url = result.get("async")
        if not audio_url:
            raise TTSError("FPT.AI TTS response did not contain an audio URL.")

        return str(audio_url)

    def download_audio(self, audio_url: str) -> bytes:
        deadline = time.monotonic() + self.config.max_wait
        attempt = 1
        last_status: Optional[int] = None

        while time.monotonic() < deadline:
            wait_time = self.config.poll_delay
            if wait_time > 0:
                remaining_wait = max(deadline - time.monotonic(), 0)
                time.sleep(min(wait_time, remaining_wait))

            try:
                response = requests.get(audio_url, timeout=self.config.timeout)
            except requests.RequestException as exc:
                if time.monotonic() >= deadline:
                    raise TTSError(f"Could not download TTS audio: {exc}") from exc

                print(f"[TTS] Audio download failed. Retrying ({attempt})...", flush=True)
                attempt += 1
                continue

            last_status = response.status_code
            if response.status_code == 200:
                if not response.content:
                    raise TTSError("Downloaded TTS audio is empty.")
                return response.content

            if response.status_code not in RETRYABLE_DOWNLOAD_STATUSES:
                raise TTSError(
                    f"Could not download TTS audio: HTTP {response.status_code}."
                )

            print(
                f"[TTS] Audio is not ready yet "
                f"(HTTP {response.status_code}). Retrying ({attempt})...",
                flush=True,
            )
            attempt += 1

        if last_status is None:
            raise TTSError("Could not download TTS audio before timeout.")

        raise TTSError(
            "Could not download TTS audio before timeout. "
            f"Last URL returned HTTP {last_status}."
        )

    def play_mp3(self, audio_data: bytes) -> None:
        try:
            from pydub import AudioSegment
        except ImportError as exc:
            raise TTSError("Missing pydub. Install dependencies from requirements.txt.") from exc

        sound = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")

        try:
            from pydub.playback import _play_with_simpleaudio

            playback = _play_with_simpleaudio(sound)
            playback.wait_done()
        except ImportError:
            from pydub.playback import play

            play(sound)

    def speak(self, text: str) -> None:
        print("[TTS] Sending answer to FPT.AI...", flush=True)
        audio_url = self.create_audio_url(text)
        print("[TTS] Downloading audio when it is ready...", flush=True)
        audio_data = self.download_audio(audio_url)
        print("[TTS] Playing audio...", flush=True)
        self.play_mp3(audio_data)


def tts_fpt_ai_v5(text_input: str) -> None:
    FPTTTSClient.from_env().speak(text_input)


if __name__ == "__main__":
    sample_text = "Xin chào, đây là phần kiểm tra chuyển văn bản thành giọng nói."
    tts_fpt_ai_v5(sample_text)
