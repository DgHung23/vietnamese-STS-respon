import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from google import genai
from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


DEFAULT_FAQ_PATH = "faq.csv"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
THRESHOLD = 0.72

DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant for Greenwich Vietnam students.
Answer in Vietnamese, clearly, politely, and concisely.
If the user's speech-to-text transcript is unclear or not a real question,
reply: "Không rõ câu hỏi, vui lòng hỏi lại."
""".strip()


class RAGError(RuntimeError):
    """Base error for the RAG/LLM responder."""


class LLMError(RAGError):
    """Raised when Gemini cannot return a usable answer."""


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class RAGConfig:
    faq_path: str = DEFAULT_FAQ_PATH
    threshold: float = THRESHOLD
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_api_key: Optional[str] = None
    max_retries: int = 3
    retry_delay: float = 2.0
    show_progress_bar: bool = False
    verbose: bool = True
    debug: bool = False


@dataclass
class RAGMatch:
    question: str
    answer: str
    score: float


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


def config_from_env(**overrides) -> RAGConfig:
    if load_dotenv is not None:
        load_dotenv()

    config = RAGConfig(
        faq_path=os.getenv("FAQ_PATH", DEFAULT_FAQ_PATH),
        threshold=_env_float("RAG_THRESHOLD", THRESHOLD),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3")),
        retry_delay=_env_float("GEMINI_RETRY_DELAY", 2.0),
        show_progress_bar=_env_bool("RAG_SHOW_PROGRESS", False),
        verbose=_env_bool("RAG_VERBOSE", True),
        debug=_env_bool("RAG_DEBUG", False),
    )

    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    return config


class HybridRAGResponder:
    """Routes a question to FAQ retrieval first, then Gemini as fallback."""

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        if load_dotenv is not None:
            load_dotenv()

        self.config = config or config_from_env()
        self.df = pd.DataFrame(columns=["question", "answer"])
        self.questions: list[str] = []
        self.embedding_model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self._client: Optional[genai.Client] = None

        self._load_faq_index()

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(message, flush=True)

    def _load_faq_index(self) -> None:
        faq_path = Path(self.config.faq_path)
        if not faq_path.exists():
            self._log(f"[RAG] FAQ file not found: {faq_path}. Gemini fallback only.")
            return

        self._log(f"[RAG] Loading FAQ data from {faq_path}...")
        df = pd.read_csv(faq_path, encoding="utf-8-sig", skipinitialspace=True)
        required_columns = {"question", "answer"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise RAGError(
                f"FAQ file must contain columns: {', '.join(sorted(required_columns))}."
            )

        df = df[["question", "answer"]].dropna()
        df["question"] = df["question"].astype(str).str.strip()
        df["answer"] = df["answer"].astype(str).str.strip()
        df = df[(df["question"] != "") & (df["answer"] != "")]
        self.df = df.reset_index(drop=True)
        self.questions = self.df["question"].tolist()

        if not self.questions:
            self._log("[RAG] FAQ file has no usable rows. Gemini fallback only.")
            return

        self._log(f"[RAG] Loading embedding model: {self.config.embedding_model}...")
        self.embedding_model = SentenceTransformer(self.config.embedding_model)

        self._log("[RAG] Creating FAQ embeddings...")
        embeddings = self.embedding_model.encode(
            self.questions,
            normalize_embeddings=True,
            show_progress_bar=self.config.show_progress_bar,
        )
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self._log(f"[RAG] Indexed {len(self.questions)} FAQ rows.")

    def _get_client(self) -> genai.Client:
        if self._client is not None:
            return self._client

        api_key = self.config.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMError("Missing Gemini API key. Set GEMINI_API_KEY in .env.")

        self._client = genai.Client(api_key=api_key)
        return self._client

    def search_faq(self, query: str) -> Optional[RAGMatch]:
        if self.index is None or self.embedding_model is None or not self.questions:
            return None

        cleaned_query = query.strip()
        if not cleaned_query:
            return None

        query_embedding = self.embedding_model.encode(
            [cleaned_query],
            normalize_embeddings=True,
        )
        query_embedding = np.array(query_embedding).astype("float32")
        scores, indices = self.index.search(query_embedding, 1)

        best_score = float(scores[0][0])
        best_idx = int(indices[0][0])
        if best_idx < 0:
            return None

        row = self.df.iloc[best_idx]
        match = RAGMatch(
            question=str(row["question"]),
            answer=str(row["answer"]),
            score=best_score,
        )

        if self.config.debug:
            print(f"[RAG] Best FAQ question: {match.question}")
            print(f"[RAG] Similarity score: {match.score:.3f}")

        return match

    def call_llm(self, user_query: str) -> str:
        prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\nUser question: {user_query.strip()}"
        delay = self.config.retry_delay

        for attempt in range(self.config.max_retries):
            try:
                response = self._get_client().models.generate_content(
                    model=self.config.gemini_model,
                    contents=prompt,
                )
                answer = (response.text or "").strip()
                if not answer:
                    raise LLMError("Gemini returned an empty answer.")
                return answer
            except Exception as exc:
                message = str(exc)
                retryable = "503" in message or "429" in message
                last_attempt = attempt >= self.config.max_retries - 1
                if retryable and not last_attempt:
                    print(
                        "[LLM] Gemini is busy. "
                        f"Retrying in {delay:.0f}s "
                        f"({attempt + 1}/{self.config.max_retries})..."
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue

                if isinstance(exc, LLMError):
                    raise
                raise LLMError(f"Error when connecting to Gemini LLM: {message}") from exc

        raise LLMError("Gemini retry loop ended without an answer.")

    def get_response(self, query: str) -> str:
        cleaned_query = query.strip()
        if not cleaned_query:
            return "Không rõ câu hỏi, vui lòng hỏi lại."

        match = self.search_faq(cleaned_query)
        if match is not None and match.score >= self.config.threshold:
            if self.config.debug:
                print("[RAG] Route: FAQ match -> CSV answer")
            return match.answer

        if self.config.debug:
            print("[RAG] Route: no FAQ match -> Gemini")
        return self.call_llm(cleaned_query)


_default_responder: Optional[HybridRAGResponder] = None


def get_default_responder() -> HybridRAGResponder:
    global _default_responder
    if _default_responder is None:
        _default_responder = HybridRAGResponder()
    return _default_responder


def call_llm(user_query: str) -> str:
    return get_default_responder().call_llm(user_query)


def get_bot_response(query: str) -> str:
    return get_default_responder().get_response(query)


def build_arg_parser() -> argparse.ArgumentParser:
    if load_dotenv is not None:
        load_dotenv()

    parser = argparse.ArgumentParser(description="Test the RAG/LLM responder.")
    parser.add_argument("--question", help="Ask one question and exit.")
    parser.add_argument(
        "--faq-path",
        default=os.getenv("FAQ_PATH", DEFAULT_FAQ_PATH),
        help="Path to the FAQ CSV file.",
    )
    parser.add_argument(
        "--rag-threshold",
        type=float,
        default=_env_float("RAG_THRESHOLD", THRESHOLD),
        help="Minimum similarity score required to use an FAQ answer.",
    )
    parser.add_argument(
        "--rag-debug",
        action="store_true",
        default=_env_bool("RAG_DEBUG", False),
        help="Print the selected FAQ match and route.",
    )
    return parser


def main() -> None:
    configure_stdio()
    args = build_arg_parser().parse_args()
    bot = HybridRAGResponder(
        config_from_env(
            faq_path=args.faq_path,
            threshold=args.rag_threshold,
            debug=args.rag_debug,
        )
    )

    if args.question:
        print(bot.get_response(args.question))
        return

    print("\n=== Greenwich Hybrid RAG Chatbot ===")
    print("Type 'exit' to quit the chat\n")

    while True:
        try:
            query = input("You: ")
        except EOFError:
            print()
            break

        if query.lower() == "exit":
            print("Goodbye!")
            break

        reply = bot.get_response(query)
        print(f"\nBot:\n{reply}")
        print("=" * 50)


if __name__ == "__main__":
    main()
