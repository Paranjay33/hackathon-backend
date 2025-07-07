import base64
import os
import tempfile
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

ULCA_USER_ID = os.getenv("ULCA_USER_ID")
ULCA_API_KEY = os.getenv("ULCA_API_KEY")

ASR_PIPELINE_ID = "64392f96daac500b55c543cd"

if not ULCA_USER_ID or not ULCA_API_KEY:
    raise RuntimeError("Missing ULCA_USER_ID or ULCA_API_KEY in .env")

app = FastAPI(title="Bhashini API Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BackendResponse(BaseModel):
    original_text: str
    translated_text: str
    final_text: str
    audio_base64: str

async def bhashini_asr_translate_tts(wav_path: Path, lang: str) -> BackendResponse:
    cfg_url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
    cfg_payload = {"pipelineId": ASR_PIPELINE_ID, "taskType": "asr"}
    cfg_headers = {
        "userID": ULCA_USER_ID,
        "ulcaApiKey": ULCA_API_KEY,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        cfg_resp = await client.post(cfg_url, json=cfg_payload, headers=cfg_headers)
        if cfg_resp.status_code != 200:
            raise HTTPException(cfg_resp.status_code, f"Failed pipeline config: {cfg_resp.text}")
        config = cfg_resp.json()

    callback_url = config["pipelineInferenceAPIEndPoint"]["callbackUrl"]
    service_id = config["pipelineInferenceAPIEndPoint"]["serviceId"]
    token_name = config["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["name"]
    token_value = config["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]

    with open(wav_path, "rb") as f:
        wav_b64 = base64.b64encode(f.read()).decode()

    inference_payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {"sourceLanguage": lang},
                    "serviceId": service_id,
                    "audioFormat": "wav",
                    "samplingRate": 16000
                }
            },
            {
                "taskType": "translation",
                "config": {
                    "language": {"sourceLanguage": lang, "targetLanguage": "en"},
                    "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4"
                }
            },
            {
                "taskType": "tts",
                "config": {
                    "language": {"sourceLanguage": "en"},
                    "serviceId": "Bhashini/IITM/TTS",
                    "gender": "female"
                }
            }
        ],
        "inputData": {"audio": [{"audioContent": wav_b64}]}
    }

    inf_headers = {
        token_name: token_value,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=120) as client:
        inf_resp = await client.post(callback_url, json=inference_payload, headers=inf_headers)
        if inf_resp.status_code != 200:
            raise HTTPException(inf_resp.status_code, f"Inference error: {inf_resp.text}")

        data = inf_resp.json()
        try:
            asr_text = data["pipelineResponse"][0]["output"][0]["source"]
            translated = data["pipelineResponse"][1]["output"][0]["target"]
            audio_b64 = data["pipelineResponse"][2]["audio"][0]["audioContent"]
        except Exception as e:
            raise HTTPException(500, f"Invalid pipeline response: {e}")

    return BackendResponse(
        original_text=asr_text,
        translated_text=translated,
        final_text=translated,
        audio_base64=audio_b64
    )

@app.post("/process-audio", response_model=BackendResponse)
async def process_audio(audio: UploadFile = File(...), language: str = Form(...)):
    suffix = Path(audio.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = Path(tmp.name)

    try:
        return await bhashini_asr_translate_tts(tmp_path, language)
    finally:
        tmp_path.unlink(missing_ok=True)
