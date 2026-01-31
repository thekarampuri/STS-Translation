from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional
import os

from pipeline.orchestrator import orchestrator
from api.schemas import TranscriptionResponse, TTSResponse, TTSInfoResponse
from config.settings import settings

router = APIRouter()

# Serve templates
templates = Jinja2Templates(directory=os.path.join(settings.BASE_DIR, "ui", "templates"))

@router.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    audio: UploadFile = File(...), 
    lang: str = Form(...), 
    target_lang: Optional[str] = Form(None)
):
    audio_bytes = await audio.read()
    result = await orchestrator.process_speech(audio_bytes, lang, target_lang)
    if "error" in result:
        return TranscriptionResponse(text="", error=result["error"])
        
    return TranscriptionResponse(
        text=result["text"],
        translated_text=result.get("translated_text", "")
    )

@router.post("/tts", response_model=TTSResponse)
async def tts(
    text: str = Form(...), 
    lang: str = Form(...), 
    gender: str = Form("male")
):
    try:
        audio_b64 = await orchestrator.generate_tts(text, lang, gender)
        if audio_b64 is None:
            return TTSResponse(error="TTS generation failed")
            
        return TTSResponse(audio=audio_b64)
    except Exception as e:
        return TTSResponse(error=str(e))

@router.get("/tts_info", response_model=TTSInfoResponse)
async def get_tts_info():
    return TTSInfoResponse(supported_languages=orchestrator.get_tts_langs())
