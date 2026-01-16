from transformers import AutoModel, AutoConfig
import json
from huggingface_hub import hf_hub_download

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

print("\nLoading Model to check vocab size...")
try:
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("Model loaded.")
    print(f"Model class: {model.__class__.__name__}")
    # Try to find vocab size
    if hasattr(model.config, "vocab_size"):
        print(f"Config vocab_size: {model.config.vocab_size}")
    
    # Check output layer size
    # Usually model.lm_head or model.ctc_head
    if hasattr(model, "lm_head"):
        print(f"lm_head out_features: {model.lm_head.out_features} (type: {type(model.lm_head)})")
    elif hasattr(model, "ctc_head"):
         print(f"ctc_head out_features: {model.ctc_head.out_features}")
    else:
        print("Could not find head")

except Exception as e:
    print("Model load failed:", e)

# Analyze vocab.json keys
try:
    print("\nReading vocab.json...")
    vocab_file = hf_hub_download(repo_id=MODEL_ID, filename="assets/vocab.json")
    with open(vocab_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Vocab keys (languages): {list(data.keys())}")
    # Check first language length
    if data:
        first_lang = list(data.keys())[0]
        print(f"Length of vocab for {first_lang}: {len(data[first_lang])}")
        
    # Check 'all' key if exists?
    if 'all' in data:
        print(f"Length of vocab for 'all': {len(data['all'])}")

except Exception as e:
    print(e)
