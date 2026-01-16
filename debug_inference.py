import torch
from transformers import AutoModel
import inspect
import json
from huggingface_hub import hf_hub_download

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
SAMPLE_RATE = 16000

def debug_inference():
    print(f"Loading model: {MODEL_ID}")
    try:
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    print("\n--- Model Inspection ---")
    print(f"Model Class: {type(model)}")
    print(f"Forward Signature: {inspect.signature(model.forward)}")
    
    # List children to see layers
    print("\nModel Modules (Top Level):")
    for name, child in model.named_children():
        print(f" - {name}: {type(child)}")

    # Check for specific methods
    for method_name in ['forward_ctc', 'decode', 'transcribe']:
        if hasattr(model, method_name):
            print(f"Found method: {method_name}")

    # Generate dummy input
    dummy_input = torch.randn(1, SAMPLE_RATE * 3).float() # 3 seconds
    
    print("\nRunning model(input, lang='hi')...")
    try:
        # Based on previous attempts, it requires 'lang'
        outputs = model(dummy_input, lang="hi")
        print(f"Output type: {type(outputs)}")
        
        if isinstance(outputs, str):
            print(f"Output (string): '{outputs}'")
        elif isinstance(outputs, torch.Tensor):
            print(f"Output (tensor) shape: {outputs.shape}")
        elif hasattr(outputs, "logits"):
            print(f"Output has logits: {outputs.logits.shape}")
        elif isinstance(outputs, (tuple, list)):
            print(f"Output is {type(outputs)} of length {len(outputs)}")
            for i, item in enumerate(outputs):
                print(f"  Item {i}: {type(item)}")
                if hasattr(item, "shape"):
                    print(f"    Shape: {item.shape}")
        else:
            print(f"Output content: {outputs}")

    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_inference()
