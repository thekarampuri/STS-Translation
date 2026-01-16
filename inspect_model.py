from transformers import AutoModel, AutoConfig
import torch
import sys

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

print("Loading model...")
try:
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"Model class: {model.__class__.__name__}")
    print(f"Model dir: {dir(model)}")
    
    # Create dummy input
    # Vocab size is 257?
    input_values = torch.randn(1, 16000).float() # 1 second audio
    # The model might expect 'input_values', 'attention_mask'
    # And maybe 'input_lengths'?
    
    print("Running forward pass with dummy input...")
    try:
        # Some models expect input_features, some input_values
        # Wav2Vec2 usually takes input_values
        outputs = model(input_values=input_values)
        print(f"Output type: {type(outputs)}")
        if hasattr(outputs, "keys"):
             print(f"Output keys: {outputs.keys()}")
        if hasattr(outputs, "logits"):
             print(f"Logits shape: {outputs.logits.shape}")
        elif hasattr(outputs, "last_hidden_state"):
             print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
             # Check if there is a head manually?
             
    except Exception as e:
        print(f"Forward failed: {e}")

except Exception as e:
    print(f"Load failed: {e}")
