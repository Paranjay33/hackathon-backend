# backend/main.py
"""
FastAPI backend for the Flutter "Transcribe & Translate" app
-----------------------------------------------------------
• Receives either AUDIO (multipart) or TEXT (JSON)
• Uses **Bhashini** only: STT, Translation, TTS
• Returns JSON with transcription, translations and base‑64 MP3

ENVIRONMENT VARIABLES (in a .env file, NOT committed):
    ULCA_USER_ID=<profile userID>
    ULCA_API_KEY=<profile ulcaApiKey>
    BHASHINI_AUTH=<long Authorization token>

Run locally:
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Environment & constants
# ---------------------------------------------------------------------------
ULCA_USER_ID: str | None = os.getenv("ULCA_USER_ID")
ULCA_API_KEY: str | None = os.getenv("ULCA_API_KEY")
BHASHINI_AUTH: str | None = os.getenv("BHASHINI_AUTH")

if not all([ULCA_USER_ID, ULCA_API_KEY, BHASHINI_AUTH]):
    raise RuntimeError("ULCA_USER_ID, ULCA_API_KEY, BHASHINI_AUTH must be set in .env")

TRANSLATE_URL = "https://bhashini.gov.in/ulca/apis/v1/translate"
TTS_URL = "https://bhashini.gov.in/ulca/apis/v1/synthesize"
# Public pipeline to access MeitY ASR (16 kHz wav mono). Replace if needed.
ASR_PIPELINE_ID = "64392f96daac500b55c543cd"


def _bhashini_headers() -> dict[str, str]:
    """Unified headers for all synchronous v1 endpoints (Translate, TTS)."""
    return {
        "userID": ULCA_USER_ID,
        "ulcaApiKey": ULCA_API_KEY,
        "Authorization": BHASHINI_AUTH,
        "Content-Type": "application/json",
    }

# ---------------------------------------------------------------------------
# FastAPI app & CORS – allow Flutter debug and production domains
# ---------------------------------------------------------------------------
app = FastAPI(title="Bhashini Backend")

origins = [
    "http://localhost:3000",  # Flutter web debug
    "http://127.0.0.1:3000",
    "*",  # TODO: restrict in prod
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class TextRequest(BaseModel):
    text: str
    language: str  # ISO 639‑1 or Bhashini code (e.g. "hi")

class BackendResponse(BaseModel):
    original_text: str
    translated_text: str
    final_text: str
    audio_base64: str

# ---------------------------------------------------------------------------
# Helper functions – Bhashini integration
# ---------------------------------------------------------------------------

async def bhashini_translate(text: str, src_lang: str, tgt_lang: str) -> str:
    payload = {
        "inputText": text,
        "inputLanguage": src_lang,
        "outputLanguage": tgt_lang,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(TRANSLATE_URL, json=payload, headers=_bhashini_headers())
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, f"Translate failed: {resp.text}")
        return resp.text.strip()


async def bhashini_tts(text: str, lang: str) -> bytes:
    payload = {"text": text, "language": lang, "voiceName": "Female1"}
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(TTS_URL, json=payload, headers=_bhashini_headers())
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, f"TTS failed: {resp.text}")
        return resp.content  # audio/mpeg (MP3)


async def bhashini_asr(wav_path: Path, lang: str) -> str:
    """ULCA pipeline flow for ASR (two‑step)."""
    # Step 1: get pipeline config (needs ULCA creds)
    cfg_url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
    cfg_payload = {"pipelineId": ASR_PIPELINE_ID, "taskType": "asr"}
    cfg_headers = {
        "userID": ULCA_USER_ID,
        "ulcaApiKey": ULCA_API_KEY,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        cfg = await client.post(cfg_url, json=cfg_payload, headers=cfg_headers)
        cfg.raise_for_status()
    c = cfg.json()
    inf_url = c["pipelineInferenceAPIEndPoint"]["callbackUrl"]
    svc_id = c["pipelineInferenceAPIEndPoint"]["serviceId"]
    key_name = c["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["name"]
    key_val = c["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]

    with open(wav_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    inference_payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {"sourceLanguage": lang},
                    "serviceId": svc_id,
                    "audioFormat": "wav",
                    "samplingRate": 16000,
                },
            }
        ],
        "inputData": {"audio": [{"audioContent": audio_b64}]},
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(inf_url, json=inference_payload, headers={key_name: key_val})
        resp.raise_for_status()
    try:
        return resp.json()["pipelineResponse"][0]["output"]
    except Exception as e:
        raise HTTPException(500, f"ASR parse error: {e}")


async def process_text_pipeline(text: str, lang: str) -> BackendResponse:
    translated = await bhashini_translate(text, lang, "English")
    # Echo back as demo (no chatbot). Replace with logic if you have NLLB etc.
    final_text = await bhashini_translate(translated, "English", lang)
    audio_bytes = await bhashini_tts(final_text, lang)
    audio_b64 = base64.b64encode(audio_bytes).decode()

    return BackendResponse(
        original_text=text,
        translated_text=translated,
        final_text=final_text,
        audio_base64=audio_b64,
    )


async def process_audio_pipeline(file: UploadFile, lang: str) -> BackendResponse:
    suffix = Path(file.filename or "audio").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        stt_text = await bhashini_asr(tmp_path, lang)
        return await process_text_pipeline(stt_text, lang)
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------

@app.post("/process-text", response_model=BackendResponse)
async def process_text(request: TextRequest):
    return await process_text_pipeline(request.text, request.language)

@app.post("/process-audio", response_model=BackendResponse)
async def process_audio(audio: UploadFile = File(...), language: str = Form(...)):
    print(f"Received audio: {audio.filename}, content_type: {audio.content_type}, language: {language}")

    #if audio.content_type not in {"audio/wav", "audio/x-wav", "audio/mpeg"}:
        #raise HTTPException(400, "Unsupported audio format. Use WAV 16 kHz or MP3.")
    return await process_audio_pipeline(audio, language)


# ---------------------------------------------------------------------------
# requirements.txt (for reference)
# ---------------------------------------------------------------------------
"""
fastapi
uvicorn[standard]
httpx
python-dotenv
aiofiles
"""
