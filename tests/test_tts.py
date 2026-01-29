import sys
import os
import shutil

# Ensure project root is in path ensuring we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tts_engine.engine import TTSEngine
import soundfile as sf

def test_tts():
    print("Initializing TTS Engine...")
    tts = TTSEngine()
    
    print("Supported languages:", tts.get_supported_languages())
    
    lang = "hi"  # Hindi
    text = "नमस्ते, यह एक परीक्षण है।" # Namaste, this is a test.
    
    print(f"Synthesizing '{text}' in {lang}...")
    try:
        wav = tts.synthesize(text, lang)
        
        if wav is not None:
            print("Synthesis successful. Shape:", wav.shape)
            sf.write("tests/output.wav", wav, 16000)
            print("Saved to tests/output.wav")
        else:
            print("Synthesis returned None.")
            
    except Exception as e:
        print(f"Synthesis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tts()
