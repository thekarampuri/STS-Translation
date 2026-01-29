import os
import torch
import logging
import numpy as np
from typing import Optional, Dict

# External dependencies (assumed installed in env)
from TTS.utils.synthesizer import Synthesizer

# Internal refactored module
from tts_engine.internal.src.inference import TextToSpeechEngine as InternalTTSEngine
from tts_engine.configs import TTSConfigResolver
from config.settings import settings
from core.device_manager import device_manager

logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self):
        self.models = {}
        self.engine = None
        self.device = device_manager.get_device()
        self.checkpoint_root = settings.TTS_CHECKPOINTS_DIR
        
    def load_language(self, lang: str):
        if lang in self.models:
            return

        lang_path = os.path.join(self.checkpoint_root, lang)
        if not os.path.exists(lang_path):
            logger.error(f"Model for language '{lang}' not found at {lang_path}")
            return
            # raise FileNotFoundError(f"Model for language '{lang}' not found at {lang_path}")

        logger.info(f"Loading TTS model for {lang} from {lang_path}...")
        
        # Resolve config paths dynamically
        tts_config = os.path.join(lang_path, "fastpitch", "config.json")
        resolved_tts_config = TTSConfigResolver.ensure_resolved_config(tts_config)
        
        vocoder_config = os.path.join(lang_path, "hifigan", "config.json")
        resolved_vocoder_config = TTSConfigResolver.ensure_resolved_config(vocoder_config)

        self.models[lang] = Synthesizer(
            tts_checkpoint=os.path.join(lang_path, "fastpitch", "best_model.pth"),
            tts_config_path=resolved_tts_config,
            tts_speakers_file=os.path.join(lang_path, "fastpitch", "speakers.pth"),
            tts_languages_file=None,
            vocoder_checkpoint=os.path.join(lang_path, "hifigan", "best_model.pth"),
            vocoder_config=resolved_vocoder_config,
            encoder_checkpoint="",
            encoder_config="",
            use_cuda=device_manager.is_cuda(),
        )
        logger.info(f"Successfully loaded {lang} Synthesizer.")
        
        # Re-initialize the internal engine
        self.engine = InternalTTSEngine(self.models, allow_transliteration=False, enable_denoiser=False)
        logger.info(f"Internal engine updated.")

    def get_supported_languages(self):
        if not os.path.exists(self.checkpoint_root):
            return []
        return [d for d in os.listdir(self.checkpoint_root) 
                if os.path.isdir(os.path.join(self.checkpoint_root, d))]

    def synthesize(self, text: str, lang: str, gender: str = "male") -> Optional[np.ndarray]:
        try:
            if lang not in self.models:
                self.load_language(lang)
            
            if not self.engine:
                logger.error("Engine not initialized.")
                return None

            wav = self.engine.infer_from_text(
                input_text=text,
                lang=lang,
                speaker_name=gender
            )
            return wav
        except Exception as e:
            logger.error(f"TTS Error for {lang}: {e}")
            return None
