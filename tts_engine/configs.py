import os
import json
import logging
import tempfile
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TTSConfigResolver:
    @staticmethod
    def resolve_path(base_path: str, path_in_config: str) -> str:
        """
        Resolves a path found in config.json relative to the checkpoint directory.
        """
        if path_in_config is None:
            return None
            
        # If it's just a filename (e.g. "speakers.pth"), join with base
        if os.path.basename(path_in_config) == path_in_config:
             return os.path.join(base_path, path_in_config)

        # check if it exists as absolute
        if os.path.isabs(path_in_config) and os.path.exists(path_in_config):
            return path_in_config
            
        # if it's absolute but doesn't exist, try to find it in base_path
        filename = os.path.basename(path_in_config)
        candidate = os.path.join(base_path, filename)
        if os.path.exists(candidate):
            return candidate
            
        # fallback for subdirs like "fastpitch/speakers.pth"
        if "/" in path_in_config or "\\" in path_in_config:
             parts = path_in_config.replace("\\", "/").split("/")
             for i in range(len(parts)):
                 suffix = os.path.join(*parts[i:])
                 candidate = os.path.join(base_path, suffix)
                 if os.path.exists(candidate):
                     return candidate
        
        return path_in_config

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        base_dir = os.path.dirname(config_path)
        
        updated = False
        if "speakers_file" in config:
            new_path = TTSConfigResolver.resolve_path(base_dir, config["speakers_file"])
            if new_path != config["speakers_file"]:
                config["speakers_file"] = new_path
                updated = True
            
        if "model_args" in config and "speakers_file" in config["model_args"]:
            new_path = TTSConfigResolver.resolve_path(base_dir, config["model_args"]["speakers_file"])
            if new_path != config["model_args"]["speakers_file"]:
                config["model_args"]["speakers_file"] = new_path
                updated = True

        return config

    @staticmethod
    def ensure_resolved_config(config_path: str) -> str:
        """
        Reads the config, resolves paths, and if changes are needed, writes to a temp file.
        Returns the path to the (possibly temp) config file.
        """
        try:
            config = TTSConfigResolver.load_config(config_path)
            # We can't easily know if it changed without comparing, 
            # but writing a temp file is safer than modifying original if we want "no manual scripts".
            # Actually, to be performant, maybe just write it always to temp?
            # Or check if we can write back to original? User said "No manual editing of config.json".
            # Safe bet: Write to a temp file in the same directory (to keep relative paths working if any other remain?)
            # Or just temp dir? Speakers file is resolved to absolute, so temp dir is fine.
            
            fd, temp_path = tempfile.mkstemp(suffix=".json", text=True)
            with os.fdopen(fd, 'w') as f:
                json.dump(config, f, indent=4)
            
            return temp_path
        except Exception as e:
            logger.error(f"Error resolving config config {config_path}: {e}")
            return config_path
