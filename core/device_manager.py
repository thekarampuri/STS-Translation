import torch
import logging

class DeviceManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"DeviceManager initialized. Using device: {cls._instance.device}")
        return cls._instance

    def get_device(self):
        return self.device

    def is_cuda(self):
        return self.device == "cuda"

device_manager = DeviceManager()
