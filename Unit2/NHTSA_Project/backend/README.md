# Backend

FastAPI service that hosts the LangGraph NHTSA complaint-analysis pipeline as
an HTTP API. Deployed on Railway; consumed by the Vercel-hosted frontend.

## What it serves

| Method | Path                | Purpose                                                     |
| ------ | ------------------- | ----------------------------------------------------------- |
| GET    | `/`                 | Service banner — handy as a manual "is the deploy alive?"   |
| GET    | `/health`           | Railway healthcheck probe                                   |
| POST   | `/run`              | Streams pipeline progress to the caller as Server-Sent Events |
| POST   | `/audio/transcribe` | Sends uploaded browser audio to OpenAI STT and returns JSON |
| POST   | `/audio/speech`     | Sends text to Deepgram TTS and returns browser-playable WAV |

`POST /run` request body:

```json
{ "user_request": "What were the most-reported safety subsystems in 2024?" }
```

`POST /run` response: `text/event-stream` of SSE frames. Each frame has a
named event and a JSON `data` payload:

```
event: started
data: {"user_request": "..."}

event: node_update
data: {"node": "classify_task", "delta": {"task_type": "retrieve"}}

event: completed
data: {"response": "...", "task_type": "retrieve", "row_count": 12, ...}
```

Errors land on the same stream as `event: error` rather than HTTP 500s, so
the frontend can render them without dropping the connection mid-stream.

## How the LangGraph pipeline runs server-side

The pipeline was originally written for an interactive CLI with audio I/O
and a human at stdin. The backend flips two process-wide flags before the
graph module is imported (see `pipeline_runner.py`):

- `set_audio_enabled(False)` — the mic/speaker code paths never run.
- `set_headless_mode(True)` — every interactive tool short-circuits:
  - `voice_ask_user` and `Ask_User` raise a hosted clarification checkpoint so
    the browser can collect the human's answer and resume the same run.
  - `code_exec` is hard-disabled (no sandbox on the public URL).
  - `confirm_csv_write` auto-declines so no CSV is ever mutated server-side.
  - `narrate_node_result` / `narrate_progress` no-op (progress is surfaced
    via the SSE stream's per-node events instead).

The local CLI never sets headless mode, so its existing behavior is unchanged.

The `/audio/*` endpoints do not re-enable the local CLI audio stack. They call
hosted audio APIs instead:

- `/audio/transcribe` accepts `multipart/form-data`, forwards the audio file to
  OpenAI `v1/audio/transcriptions`, and sets `chunking_strategy=auto` so
  server-side VAD handles speech boundaries.
- `/audio/speech` sends text to Deepgram Speak v1 using the existing Deepgram
  default voice/model preset from `Runtime_Options`, then wraps the returned
  linear16 PCM bytes as WAV so the frontend can play the response.

This keeps Whisper, Silero VAD, torch, ffmpeg, Kokoro, sounddevice, and local
speaker playback out of the Railway runtime.

## CORS

`main.py` allows the production Vercel URL, `localhost:3000`, and any
`*.vercel.app` preview deploy. CORS is a server-side allowlist of *callers*,
not of where this server is hosted — adding the backend URL itself to that list
would do nothing.

## Railway settings

| Setting          | Value                                              |
| ---------------- | -------------------------------------------------- |
| Root Directory   | `Unit2/NHTSA_Project`                              |
| Runtime          | Python                                             |
| Python Version   | `3.11` (pinned via `backend/.python-version`)      |
| Build Command    | `pip install -r requirements.txt`                  |
| Start Command    | `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT` |

`backend/.python-version` keeps Railway on the same major/minor as the local
`.venv311` (currently `Python 3.11.14`). Without it, Railway can default to the
latest CPython release at service-creation time (currently 3.14.x), which
breaks ML/data wheels that haven't published 3.14 builds yet. Pinning to
`3.11` (no patch) lets Railway pick the latest 3.11.x — stable and ABI
compatible with the local venv.

Environment variables on Railway must include the same keys the local `.env`
provides (e.g. `OPENAI_API_KEY`, `INCEPTION_API_KEY`, `DEEPGRAM_TTS_KEY`).
`OPENAI_API_KEY` is used for STT; `DEEPGRAM_TTS_KEY` is used for TTS.
`load_dotenv()` is a no-op on Railway since the variables are already in
`os.environ` before Python starts.

## Run it locally

```
cd Unit2/NHTSA_Project/backend
/path/to/.venv311/bin/python -m uvicorn main:app --reload --port 8000
```

Smoke tests:

```
curl http://localhost:8000/health
curl -N -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"user_request":"give me 3 complaints involving the brake system"}'
curl -X POST http://localhost:8000/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text":"Here is the NHTSA summary."}' \
  --output response.wav
```

`-N` disables curl's output buffering so SSE events appear as they arrive.

## Files

| Path                  | Purpose                                                   |
| --------------------- | --------------------------------------------------------- |
| `main.py`             | FastAPI app, CORS, route handlers                         |
| `audio_io.py`         | Hosted STT/TTS API adapter for browser audio              |
| `pipeline_runner.py`  | LangGraph adapter + SSE event encoder                     |
| `requirements.txt`    | Hand-curated install set (no macOS-only audio packages)   |
