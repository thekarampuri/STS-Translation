import logging
import base64
import io
import soundfile as sf
from typing import Optional, Dict

from pipeline.stt_engine import stt_engine
from pipeline.mt_engine import mt_engine
from tts_engine.engine import TTSEngine
from core.audio import convert_webm_to_wav
from config.settings import settings

logger = logging.getLogger(__name__)

class STSOrchestrator:
    def __init__(self):
        self.tts = TTSEngine()

    def process_speech(self, audio_bytes: bytes, src_lang: str, tgt_lang: Optional[str] = None):
        """
        Full pipeline: Audio -> STT -> Translate -> TTS
        Returns: { "text": ..., "translated_text": ..., "audio": ... }
        """
        # 1. Convert Audio
        wav_data = convert_webm_to_wav(audio_bytes, settings.SAMPLE_RATE)
        if len(wav_data) == 0:
            return {"error": "Empty or invalid audio"}

        # 2. STT
        transcribed_text = stt_engine.transcribe(wav_data, src_lang)
        logger.info(f"STT Output: {transcribed_text}")
        
        result = {
            "text": transcribed_text,
            "translated_text": ""
        }
        
        # 3. Translation
        if tgt_lang and tgt_lang != src_lang and transcribed_text:
            translated_text = mt_engine.translate(transcribed_text, src_lang, tgt_lang)
            result["translated_text"] = translated_text
            logger.info(f"MT Output: {translated_text}")
        
        return result

    def generate_tts(self, text: str, lang: str, gender: str) -> Optional[str]:
        """
        Returns base64 encoded WAV
        """
        audio_arr = self.tts.synthesize(text, lang, gender)
        if audio_arr is None:
            return None
            
        byte_io = io.BytesIO()
        sf.write(byte_io, audio_arr, 16000, format='WAV')
        byte_io.seek(0)
        return base64.b64encode(byte_io.read()).decode('utf-8')
        
    def get_tts_langs(self):
        return self.tts.get_supported_languages()

orchestrator = STSOrchestrator()
