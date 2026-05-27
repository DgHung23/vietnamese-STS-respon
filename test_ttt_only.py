from ttt import GeminiError, answer_question


def main() -> None:
    transcript = input("Enter your question: ").strip()
    try:
        answer = answer_question(transcript)
        print(f"Answer: {answer}", flush=True)
    except GeminiError as exc:
        print(f"LLM error: {exc}", flush=True)

if __name__ == "__main__":
    main()