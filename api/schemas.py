from pydantic import BaseModel
from typing import Optional, List, Any

class TranscriptionResponse(BaseModel):
    text: str
    translated_text: Optional[str] = ""
    error: Optional[str] = None

class TTSResponse(BaseModel):
    audio: Optional[str] = None
    error: Optional[str] = None

class TTSInfoResponse(BaseModel):
    supported_languages: List[str]
