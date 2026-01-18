import os
import sys
import torch
import numpy as np
import soundfile as sf

# Add current dir to sys.path
sys.path.append(os.getcwd())

from tts_utils import tts_engine

def test_tts(text, lang, filename):
    print(f"Testing TTS for {lang}: '{text}'")
    try:
        wav = tts_engine.synthesize(text, lang, gender="male")
        if wav is not None:
            print(f"Successfully synthesized audio. Shape: {wav.shape}")
            sf.write(filename, wav, 16000)
            print(f"Saved to {filename}")
        else:
            print("Synthesis failed (returned None)")
    except Exception as e:
        print(f"Error during synthesis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tts("नमस्ते, यह एक परीक्षण है।", "hi", "test_hi.wav")
    test_tts("Hello, this is a test.", "en", "test_en.wav")
