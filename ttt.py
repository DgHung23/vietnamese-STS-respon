import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional
import sys

try:
    import rag  # Import file rag.py làm một module xử lý dữ liệu nền
except ImportError:
    print("Error: Could not find 'rag.py' in the same directory. Please place them together.")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


"""GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)"""

DEFAULT_SYSTEM_PROMPT = (
    "Bạn là một AI trả lời câu hỏi sau khi STT. "
    "Nếu văn bản có từ ngữ lạ, bị sai do nhận diện giọng nói, hoặc không giống một câu hỏi rõ ràng, "
    "hãy trả lời ngắn gọn: 'Không rõ câu hỏi, vui lòng hỏi lại.' "
    "Nếu câu hỏi rõ ràng, hãy trả lời bằng tiếng Việt, ngắn gọn và trực tiếp."
)


"""class GeminiError(RuntimeError):
    Raised when Gemini cannot return a usable answer.


def _extract_answer(data: dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        raise GeminiError("Gemini response has no candidates.")

    parts = candidates[0].get("content", {}).get("parts") or []
    text_parts = [part.get("text", "") for part in parts if part.get("text")]
    answer = "\n".join(text_parts).strip()
    if not answer:
        raise GeminiError("Gemini response has no text answer.")

    return answer"""


"""def answer_question(
    question_text: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> str:
    question_text = question_text.strip()
    if not question_text:
        return "Không rõ câu hỏi, vui lòng hỏi lại."

    key = api_key or GEMINI_API_KEY
    if not key:
        raise GeminiError("Missing Gemini API key. Set GEMINI_API_KEY in environment.")

    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": question_text}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
        },
    }

    request = urllib.request.Request(
        GEMINI_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise GeminiError(f"Gemini API error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise GeminiError(f"Could not connect to Gemini API: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GeminiError("Gemini API returned invalid JSON.") from exc

    return _extract_answer(data)"""




try:
    # import the tts function from app.py if it exists, otherwise set to None
    from app import tts_fpt_ai_v5
except ImportError:
    print("The app.py file could not be found. Pls check the folder path again!")
    tts_fpt_ai_v5 = None


def main() -> None:
    # type question from user input
    question = input("Question: ")
    
    # send question to Gemini and get the answer
    # answer = answer_question(question) maybe not need
    answer = rag.get_bot_response(question)
    print(f"Answer: {answer}")
    
    # if have tts function and an answer, convert the answer to speech
    if tts_fpt_ai_v5 and answer:
        print("\nConverting answer to speech, pls wait...")
        # run tts function with the answer text to get audio output
        tts_fpt_ai_v5(text_input=answer)


if __name__ == "__main__":
    main()
