# main.py

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
import traceback

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()

ULCA_USER_ID = os.getenv("ULCA_USER_ID")
ULCA_API_KEY = os.getenv("ULCA_API_KEY")
BHASHINI_AUTH = os.getenv("BHASHINI_AUTH")

if not all([ULCA_USER_ID, ULCA_API_KEY, BHASHINI_AUTH]):
    raise RuntimeError("Missing API keys in .env")

ASR_PIPELINE_ID = "64392f96daac500b55c543cd"
TRANSLATE_URL = "https://bhashini.gov.in/ulca/apis/v1/translate"
TTS_URL = "https://bhashini.gov.in/ulca/apis/v1/synthesize"
CFG_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"

def _bhashini_headers() -> dict[str, str]:
    return {
        "userID": ULCA_USER_ID,
        "ulcaApiKey": ULCA_API_KEY,
        "Authorization": BHASHINI_AUTH,
        "Content-Type": "application/json",
    }

# ---------------------------------------------------------------------------
# FastAPI app and CORS setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Bhashini Voice App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class TextRequest(BaseModel):
    text: str
    language: str

class BackendResponse(BaseModel):
    original_text: str
    translated_text: str
    final_text: str
    audio_base64: str

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
async def bhashini_translate(text: str, src_lang: str, tgt_lang: str) -> str:
    payload = {
        "inputText": text,
        "inputLanguage": src_lang,
        "outputLanguage": tgt_lang,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(TRANSLATE_URL, headers=_bhashini_headers(), json=payload)
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"Translate error: {r.text}")
        return r.text.strip()

async def bhashini_tts(text: str, lang: str) -> bytes:
    payload = {
        "text": text,
        "language": lang,
        "voiceName": "Female1"
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(TTS_URL, headers=_bhashini_headers(), json=payload)
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"TTS error: {r.text}")
        return r.content

async def bhashini_asr(wav_path: Path, lang: str) -> str:
    cfg_payload = {
        "pipelineId": ASR_PIPELINE_ID,
        "taskType": "asr"
    }
    cfg_headers = {
        "userID": ULCA_USER_ID,
        "ulcaApiKey": ULCA_API_KEY,
    }

    async with httpx.AsyncClient(timeout=20) as client:
        cfg = await client.post(CFG_URL, headers=cfg_headers, json=cfg_payload)
        cfg.raise_for_status()

    data = cfg.json()
    inf_url = data["pipelineInferenceAPIEndPoint"]["callbackUrl"]
    svc_id = data["pipelineInferenceAPIEndPoint"]["serviceId"]
    key_name = data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["name"]
    key_val = data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]

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
                    "samplingRate": 16000
                }
            }
        ],
        "inputData": {
            "audio": [{"audioContent": audio_b64}]
        }
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(inf_url, headers={key_name: key_val}, json=inference_payload)
        r.raise_for_status()
        try:
            return r.json()["pipelineResponse"][0]["output"]
        except Exception as e:
            raise HTTPException(500, f"ASR failed to parse output: {e}")

# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------
async def process_text_pipeline(text: str, lang: str) -> BackendResponse:
    translated = await bhashini_translate(text, lang, "en")
    final_text = await bhashini_translate(translated, "en", lang)
    audio_bytes = await bhashini_tts(final_text, lang)
    return BackendResponse(
        original_text=text,
        translated_text=translated,
        final_text=final_text,
        audio_base64=base64.b64encode(audio_bytes).decode()
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
# API routes
# ---------------------------------------------------------------------------
@app.post("/process-text", response_model=BackendResponse)
async def process_text(request: TextRequest):
    return await process_text_pipeline(request.text, request.language)

@app.post("/process-audio", response_model=BackendResponse)
async def process_audio(audio: UploadFile = File(...), language: str = Form(...)):
    try:
        print(f"Received audio: {audio.filename}, content_type: {audio.content_type}, language: {language}")
        return await process_audio_pipeline(audio, language)
    except Exception as e:
        print("❌ Exception in /process-audio:", e)
        traceback.print_exc()
        raise HTTPException(500, "Internal Server Error")
