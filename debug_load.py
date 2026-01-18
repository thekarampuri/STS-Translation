import sys
import os
from TTS.utils.synthesizer import Synthesizer

lang_path = r"E:\Projects\STS-Translation\Indic-TTS\inference\checkpoints\en"
print("Starting Synthesizer load...")
try:
    s = Synthesizer(
        tts_checkpoint=os.path.join(lang_path, "fastpitch", "best_model.pth"),
        tts_config_path=os.path.join(lang_path, "fastpitch", "config.json"),
        tts_speakers_file=os.path.join(lang_path, "fastpitch", "speakers.pth"),
        tts_languages_file=None,
        vocoder_checkpoint=os.path.join(lang_path, "hifigan", "best_model.pth"),
        vocoder_config=os.path.join(lang_path, "hifigan", "config.json"),
        encoder_checkpoint="",
        encoder_config="",
        use_cuda=False,
    )
    print("Synthesizer loaded successfully!")
    
    import numpy as np
    import soundfile as sf
    
    print("Starting synthesis...")
    wav = s.tts("Hello, this is a test.", speaker_name="male")
    print(f"Synthesis finished. Wav type: {type(wav)}, length: {len(wav) if wav is not None else 'None'}")
    
    if wav is not None:
        sf.write("debug_en.wav", wav, 22050)
        print("Saved to debug_en.wav")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
print("End of script reached.")
