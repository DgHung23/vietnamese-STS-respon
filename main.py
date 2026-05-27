from stt import list_input_devices, parse_args, settings_from_args, stream_realtime_transcripts
from ttt import GeminiError, answer_question


def main() -> None:
    args = parse_args()

    if args.list_devices:
        list_input_devices()
        return

    settings = settings_from_args(args)

    print("Listening for Vietnamese speech. Press Ctrl+C to stop.")
    print(f"Language: {settings.language_code}, model: {settings.model}")

    try:
        for transcript in stream_realtime_transcripts(settings):
            print(f"\nSTT: {transcript}", flush=True)
            try:
                answer = answer_question(transcript)
            except GeminiError as exc:
                print(f"LLM error: {exc}", flush=True)
                continue

            print(f"AI: {answer}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()