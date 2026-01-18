import os
import sys
import torch
import numpy as np
from typing import Optional

# Add Indic-TTS/inference to sys.path to allow imports from its src
INDIC_TTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "Indic-TTS", "inference"))
if INDIC_TTS_DIR not in sys.path:
    sys.path.append(INDIC_TTS_DIR)

from src.inference import TextToSpeechEngine
from TTS.utils.synthesizer import Synthesizer

class TTSEngine:
    def __init__(self):
        self.models = {}
        self.engine = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_root = os.path.join(INDIC_TTS_DIR, "checkpoints")

    def load_language(self, lang: str):
        if lang in self.models:
            return
        
        lang_path = os.path.join(self.checkpoint_root, lang)
        if not os.path.exists(lang_path):
            raise FileNotFoundError(f"Model for language '{lang}' not found at {lang_path}")

        print(f"Loading TTS model for {lang} from {lang_path}...")
        print(f"Using device: {self.device}")
        self.models[lang] = Synthesizer(
            tts_checkpoint=os.path.join(lang_path, "fastpitch", "best_model.pth"),
            tts_config_path=os.path.join(lang_path, "fastpitch", "config.json"),
            tts_speakers_file=os.path.join(lang_path, "fastpitch", "speakers.pth"),
            tts_languages_file=None,
            vocoder_checkpoint=os.path.join(lang_path, "hifigan", "best_model.pth"),
            vocoder_config=os.path.join(lang_path, "hifigan", "config.json"),
            encoder_checkpoint="",
            encoder_config="",
            use_cuda=(self.device == "cuda"),
        )
        print(f"Successfully loaded {lang} Synthesizer.")
        
        # Re-initialize the engine with updated models
        self.engine = TextToSpeechEngine(self.models, allow_transliteration=False)
        print(f"Successfully initialized TextToSpeechEngine for {lang}.")

    def get_supported_languages(self):
        """Returns a list of languages that have checkpoints available."""
        if not os.path.exists(self.checkpoint_root):
            return []
        return [d for d in os.listdir(self.checkpoint_root) 
                if os.path.isdir(os.path.join(self.checkpoint_root, d))]

    def synthesize(self, text: str, lang: str, gender: str = "male") -> Optional[np.ndarray]:
        """
        Synthesize text to audio.
        Returns: numpy array of the audio.
        """
        try:
            if lang not in self.models:
                self.load_language(lang)
            
            if not self.engine:
                self.engine = TextToSpeechEngine(self.models, allow_transliteration=False)

            # Indic-TTS inference/src/inference.py uses Speaker Names like 'male' or 'female'
            # as defined in the speakers.pth or models.
            wav = self.engine.infer_from_text(
                input_text=text,
                lang=lang,
                speaker_name=gender
            )
            return wav
        except Exception as e:
            print(f"TTS Error for {lang}: {e}")
            return None

# Global instance
tts_engine = TTSEngine()
