import os
import json

def fix_configs(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if "config.json" in files:
            config_path = os.path.join(root, "config.json")
            print(f"Checking {config_path}...")
            
            with open(config_path, "r") as f:
                try:
                    config = json.load(f)
                except Exception as e:
                    print(f"Error loading {config_path}: {e}")
                    continue
            
            updated = False
            # Check for speakers_file in the root of config
            if "speakers_file" in config and config["speakers_file"] is not None:
                if "models/v1/" in config["speakers_file"] or config["speakers_file"] == "speakers.pth":
                    config["speakers_file"] = os.path.join(root, "speakers.pth")
                    updated = True
            
            # Check for speakers_file in model_args
            if "model_args" in config and "speakers_file" in config["model_args"]:
                if config["model_args"]["speakers_file"] is not None:
                    if "models/v1/" in config["model_args"]["speakers_file"] or config["model_args"]["speakers_file"] == "speakers.pth":
                        config["model_args"]["speakers_file"] = os.path.join(root, "speakers.pth")
                        updated = True
            
            if updated:
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)
                print(f"Updated {config_path}")

if __name__ == "__main__":
    checkpoints_dir = os.path.abspath("Indic-TTS/inference/checkpoints")
    fix_configs(checkpoints_dir)
