from transformers import AutoModel
import torch

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

def deep_inspect():
    print("Loading model...")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"Model class: {type(model)}")
    
    # Check if it has an inner model
    if hasattr(model, 'model'):
        print(f"Inner model type: {type(model.model)}")
    
    # Check all methods
    methods = [m for m in dir(model) if not m.startswith('_')]
    print(f"Methods: {methods}")
    
    # Try to see if it has a way to get logits
    # Maybe forward(..., decoding=None)?
    try:
        dummy_input = torch.randn(1, 16000).float()
        print("\nCalling forward with decoding=None...")
        out = model(dummy_input, lang="hi", decoding=None)
        print(f"Output type with decoding=None: {type(out)}")
        if hasattr(out, "shape"):
            print(f"Shape: {out.shape}")
        elif isinstance(out, (list, tuple)):
            print(f"Length: {len(out)}")
    except Exception as e:
        print(f"Failed with decoding=None: {e}")

if __name__ == "__main__":
    deep_inspect()
