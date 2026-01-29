import gc
import torch
import logging
from typing import Optional, Any, Dict

from core.device_manager import device_manager

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, key: str, loader_func, *args, **kwargs):
        """
        Loads a model if not already loaded.
        loader_func: Function that returns the model
        """
        if key in self.models:
            return self.models[key]
            
        logger.info(f"Loading model: {key}")
        try:
            model = loader_func(*args, **kwargs)
            self.models[key] = model
            logger.info(f"Model {key} loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {key}: {e}")
            raise e

    def get_model(self, key: str) -> Optional[Any]:
        return self.models.get(key)
        
    def unload_model(self, key: str):
        if key in self.models:
            logger.info(f"Unloading model: {key}")
            del self.models[key]
            self.clear_cache()
            
    def clear_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

model_manager = ModelManager.get_instance()
