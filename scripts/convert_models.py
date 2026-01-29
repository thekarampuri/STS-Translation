import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download

def convert_nllb():
    model_id = "facebook/nllb-200-distilled-600M"
    print(f"Downloading/Locating model: {model_id}")
    try:
        path = snapshot_download(
            model_id, 
            local_dir="nllb_download", 
            local_dir_use_symlinks=False
        )
        print(f"Model path: {path}")
    except Exception as e:
        print(f"Snapshot download failed: {e}")
        return

    bin_path = os.path.join(path, "pytorch_model.bin")
    if not os.path.exists(bin_path):
        print("No pytorch_model.bin file found!")
        if os.path.exists(os.path.join(path, "model.safetensors")):
            print("model.safetensors already exists!")
        return

    print("Loading state dict manually...")
    try:
        state_dict = torch.load(bin_path, map_location="cpu") # weights_only=False default in older torch
        print("State dict loaded successfully!")
    except Exception as e:
        print(f"Manual load failed: {e}")
        return

    print("Initializing model from config...")
    try:
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_config(config)
        model.load_state_dict(state_dict)
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        print("Saving model as safetensors...")
        output_dir = "nllb-safe"
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    except Exception as e:
        print(f"Conversion failed: {e}")

if __name__ == "__main__":
    convert_nllb()
