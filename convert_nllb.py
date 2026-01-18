import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import snapshot_download

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
    # Try creating directory if it fails?
    exit(1)

bin_path = os.path.join(path, "pytorch_model.bin")
if not os.path.exists(bin_path):
    print("No pytorch_model.bin file found!")
    # Check if safetensors exists?
    if os.path.exists(os.path.join(path, "model.safetensors")):
        print("model.safetensors already exists! Why did it fail?")
    exit(1)

print("Loading state dict manually...")
try:
    # Try loading with weights_only=False as the pickle might depend on it
    # We trust this model from Facebook
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
    print("State dict loaded successfully!")
except Exception as e:
    print(f"Manual load failed: {e}")
    exit(1)

print("Initializing model from config...")
try:
    # Load config and create model
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_config(config)
    
    # Load state dict
    model.load_state_dict(state_dict)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Saving model as safetensors...")
    output_dir = "nllb-safe"
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
except Exception as e:
    print(f"Conversion failed: {e}")
    exit(1)
