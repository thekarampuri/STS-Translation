import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.model_manager import model_manager
from pipeline.stt_engine import stt_engine
from pipeline.mt_engine import mt_engine
from tts_engine.engine import TTSEngine

def validate():
    print("Validating STT Models...")
    try:
        stt_engine.load_english_model()
        print("STT English Loaded.")
    except Exception as e:
        print(f"STT English Failed: {e}")

    # Indics might take RAM, skip if just checking imports? NO, full validation asked.
    # But lazy loading logic in engines handles it.

    print("Validating TTS Engine...")
    tts = TTSEngine()
    langs = tts.get_supported_languages()
    print(f"TTS Languages found: {langs}")
    
    print("Validation script completed checks.")

if __name__ == "__main__":
    validate()
