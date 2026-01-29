import os

class Settings:
    # App
    APP_TITLE: str = "Indic STT & Translate & TTS"
    HOST: str = "127.0.0.1"
    PORT: int = 8001
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    @property
    def TTS_CHECKPOINTS_DIR(self) -> str:
        # Default to inside core Indic-TTS folder or user specified
        env_val = os.getenv("TTS_CHECKPOINTS_DIR")
        if env_val: return env_val
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Indic-TTS", "inference", "checkpoints")
    
    # Models
    STT_MODEL_ID: str = os.getenv("STT_MODEL_ID", "ai4bharat/indic-conformer-600m-multilingual")
    STT_EN_MODEL_ID: str = os.getenv("STT_EN_MODEL_ID", "openai/whisper-tiny")
    MT_MODEL_PATH: str = os.getenv("MT_MODEL_PATH", "./nllb-safe")
    
    # Audio
    SAMPLE_RATE: int = 16000

settings = Settings()
