import torch
import logging
from transformers import AutoModel, pipeline
from config.settings import settings
from core.device_manager import device_manager
from core.model_manager import model_manager

logger = logging.getLogger(__name__)

class STTEngine:
    def __init__(self):
        self.device = device_manager.get_device()
        
    def load_indic_model(self):
        logger.info(f"Loading Indic STT model from {settings.STT_MODEL_ID}...")
        model = AutoModel.from_pretrained(settings.STT_MODEL_ID, trust_remote_code=True)
        if device_manager.is_cuda():
            model = model.to("cuda")
        model.eval()
        return model

    def load_english_model(self):
        logger.info(f"Loading English STT model...")
        device_id = 0 if device_manager.is_cuda() else -1
        return pipeline("automatic-speech-recognition", model=settings.STT_EN_MODEL_ID, device=device_id)

    def transcribe(self, audio_data: torch.Tensor, lang: str) -> str:
        """
        audio_data: Tensor of shape (1, samples) or numpy array?
        Indic model expects tensor. Whisper expects numpy.
        """
        try:
            if lang == "en":
                model = model_manager.load_model("stt_en", self.load_english_model)
                # Ensure input is numpy for pipeline
                if isinstance(audio_data, torch.Tensor):
                    audio_numpy = audio_data.cpu().numpy().squeeze()
                else:
                    audio_numpy = audio_data
                
                res = model(audio_numpy)
                return res["text"].strip()
            else:
                model = model_manager.load_model("stt_indic", self.load_indic_model)
                
                # Ensure input is tensor
                if not isinstance(audio_data, torch.Tensor):
                     audio_data = torch.from_numpy(audio_data).float()
                
                if audio_data.dim() == 1:
                    audio_data = audio_data.unsqueeze(0)
                    
                if device_manager.is_cuda():
                    audio_data = audio_data.to("cuda")
                
                with torch.inference_mode():
                    transcription = model(audio_data, lang=lang)
                
                return transcription.replace('▁', ' ').strip()
        except Exception as e:
            logger.error(f"STT Error: {e}")
            return ""

stt_engine = STTEngine()
