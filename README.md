# Vietnamese STT Response

This application listens to Vietnamese speech from a microphone, converts the speech to text with Google Cloud Speech-to-Text, then sends the transcript to Gemini to generate a short Vietnamese response.

In short, the flow is:

```text
Microphone -> Google Speech-to-Text -> Transcript -> Gemini -> Response in the terminal
```

## Key Features

- Real-time Vietnamese speech recognition.
- Automatically captures the final transcript after the user finishes speaking a sentence or phrase.
- Sends the transcript to Gemini for a response.
- Can list available input microphones and select a specific device.
- Supports configuring the language, STT model, sample rate, streaming session duration, and Gemini model through environment variables or command-line arguments.

## Prerequisites

Prepare the following before running the application:

- Python 3.10 or newer.
- A working microphone on your machine.
- A Google Cloud account with the Speech-to-Text API enabled.
- Google Cloud Application Default Credentials.
- A Gemini API key.
- The Python packages listed in `requirements.txt`.

## Installation

### 1. Create a Python Environment

Using a virtual environment is recommended to avoid conflicts with system packages.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

The application uses these libraries:

- `google-cloud-speech`: calls Google Cloud Speech-to-Text.
- `PyAudio`: reads audio from the microphone.
- `python-dotenv`: automatically loads environment variables from a `.env` file.

If installing `PyAudio` fails, the usual cause is a missing system audio library.

On Windows, you can try:

```powershell
pip install PyAudio
```

On macOS, PortAudio is usually required:

```bash
brew install portaudio
pip install PyAudio
```

On Ubuntu/Debian:

```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install PyAudio
```

## Google Cloud Speech-to-Text Configuration

The application uses Google Cloud Application Default Credentials. If ADC is not set up on your machine yet, run:

```bash
gcloud auth application-default login
```

If you need to select a Google Cloud project:

```bash
gcloud config set project YOUR_PROJECT_ID
```

Make sure the selected project has the Speech-to-Text API enabled.

## Gemini Configuration

The application reads the Gemini API key from the `GEMINI_API_KEY` environment variable. The most convenient approach is to create a `.env` file based on `.env.example`, then fill in your real key.

Example configuration:

```env
STT_LANGUAGE_CODE=vi-VN
STT_MODEL=latest_long
STT_SAMPLE_RATE=16000
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=gemini-2.5-flash
```

Do not share or commit a real API key to Git.

## Configuration Variables

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `STT_LANGUAGE_CODE` | `vi-VN` | Language code used for Google Speech-to-Text. |
| `STT_MODEL` | `latest_long` | Speech-to-Text V1 model. |
| `STT_SAMPLE_RATE` | `16000` | Microphone sample rate in Hz. |
| `STT_STREAMING_LIMIT` | `240` | Maximum number of seconds for each streaming session before reconnecting. |
| `STT_DEVICE_INDEX` | Not set | Index of the microphone to use. |
| `GEMINI_API_KEY` | None | API key used to call Gemini. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model used to generate responses. |

## Running the Full Application

After installation and configuration:

```bash
python main.py
```

When running, the terminal shows:

```text
Listening for Vietnamese speech. Press Ctrl+C to stop.
Language: vi-VN, model: latest_long
```

Speak into the microphone. When Google Speech-to-Text returns a final transcript, the program prints:

```text
STT: the content you just said
AI: the response from Gemini
```

Stop the program with `Ctrl+C`.

## Selecting an Input Microphone

If your machine has multiple microphones, first list the available devices:

```bash
python main.py --list-devices
```

Or:

```bash
python stt.py --list-devices
```

The output will look like:

```text
Input devices:
  0: Microphone ...
  1: Headset ...
```

Then run the application with `--device-index`:

```bash
python main.py --device-index 1
```

Or set it permanently in `.env`:

```env
STT_DEVICE_INDEX=1
```

## Command-Line Arguments

`main.py` and `stt.py` share the same STT configuration arguments:

```bash
python main.py --language-code vi-VN --model latest_long --sample-rate 16000
```

Available arguments:

| Argument | Meaning |
| --- | --- |
| `--language-code` | BCP-47 language code. The default is `vi-VN`. |
| `--model` | Google Speech-to-Text V1 model. The default is `latest_long`. |
| `--sample-rate` | Microphone sample rate. The default is `16000`. |
| `--device-index` | Selects a microphone by index. |
| `--streaming-limit` | Number of seconds for each streaming session before automatically reconnecting. |
| `--no-interim` | Does not request interim results from the API. |
| `--list-devices` | Lists input microphones and exits. |

Note: the main flow only processes completed transcripts marked as `is_final`. Interim results can be requested in the API configuration, but Gemini responses are generated only after a final transcript is available.

## Running Individual Parts

### Test Speech-to-Text Only

```bash
python stt.py
```

This mode only listens to the microphone and prints transcripts to the terminal. It does not call Gemini.

### Test Gemini Only

```bash
python test_ttt_only.py
```

Enter a question in the terminal. The program sends the question to Gemini and prints the response.

You can also run it directly:

```bash
python ttt.py
```

## How the Program Works

### `main.py`

This is the main application entry point. It:

1. Reads command-line arguments and environment variables.
2. If `--list-devices` is provided, prints the microphone list and exits.
3. Opens the microphone listening stream.
4. For each final transcript received from STT, sends the transcript to Gemini.
5. Prints both the transcript and the response to the terminal.
6. If Gemini fails, prints the error and continues listening for the next sentence.

### `stt.py`

This file handles the full Speech-to-Text flow:

- Opens the microphone with `PyAudio`.
- Records mono 16-bit PCM audio.
- Splits audio into small chunks, with a default chunk length of 0.1 seconds.
- Streams audio to Google Cloud Speech-to-Text.
- Filters and yields final transcripts while ignoring interim results.
- Automatically reconnects when the streaming session reaches its time limit.

Important values:

- `SAMPLE_RATE = 16000`
- `CHUNK_SIZE = SAMPLE_RATE / 10`
- `STREAMING_LIMIT_SECONDS = 240`

### `ttt.py`

This file calls Gemini:

- Reads `GEMINI_API_KEY` and `GEMINI_MODEL` from the environment.
- Creates a request to the Gemini `generateContent` API endpoint.
- Sends the user's transcript to Gemini.
- Uses a system prompt that asks Gemini to answer briefly in Vietnamese.
- If the question is unclear, asks Gemini to reply that the question is unclear and should be asked again.
- If the API returns an error, raises `GeminiError` so `main.py` can handle it.

### `test_ttt_only.py`

This small file tests the Gemini part without requiring a microphone or Google Speech-to-Text.

## Common Usage Examples

Run the application with the default configuration:

```bash
python main.py
```

Run with microphone number 2:

```bash
python main.py --device-index 2
```

Run with a 120-second streaming limit:

```bash
python main.py --streaming-limit 120
```

Run with a different STT model:

```bash
python main.py --model latest_short
```

Check the microphone list:

```bash
python main.py --list-devices
```

Test Gemini only:

```bash
python test_ttt_only.py
```

## Troubleshooting

### Missing Gemini API Key

The message may look like:

```text
LLM error: Missing Gemini API key.
```

How to fix it:

- Check that your `.env` file contains `GEMINI_API_KEY`.
- Make sure the terminal is running in the correct project directory.
- If you are not using `.env`, export the environment variable before running the application.

### Google Speech-to-Text Credentials Error

If Google Cloud reports an authentication error, run:

```bash
gcloud auth application-default login
```

Then check the selected project:

```bash
gcloud config get-value project
```

### Microphone Is Not Detected

Try listing devices:

```bash
python main.py --list-devices
```

Then select the correct device:

```bash
python main.py --device-index YOUR_DEVICE_INDEX
```

Also check the operating system microphone permissions.

### PyAudio Cannot Be Installed

This is usually an operating system environment issue, not a code issue. Install PortAudio or use a Python/PyAudio version that matches your operating system.

### Gemini Returns an HTTP Error

Common causes:

- The API key is incorrect or does not have permission.
- The model in `GEMINI_MODEL` is invalid.
- The machine does not have an internet connection.
- The request is blocked by quota limits.

## Security Notes

- Do not put real API keys in public documentation.
- Do not share a `.env` file that contains real keys.
- If a key has been exposed, revoke the old key and create a new one in Google AI Studio or Google Cloud Console.

## Recommended Usage Flow

1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Log in to Google Application Default Credentials.
4. Create `.env` and fill in the required configuration.
5. Run `python main.py --list-devices` to choose a microphone if needed.
6. Run `python main.py`.
7. Speak a Vietnamese question into the microphone and read the response in the terminal.
