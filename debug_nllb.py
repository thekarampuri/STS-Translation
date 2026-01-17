import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MT_MODEL_ID = "facebook/nllb-200-distilled-600M"

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MT_MODEL_ID)
    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL_ID)
    model.eval()
    
    print("Testing lang code conversion...")
    # Try different ways to get the lang id
    if hasattr(tokenizer, "lang_code_to_id"):
        print(f"Found lang_code_to_id: {tokenizer.lang_code_to_id.get('hin_Deva')}")
    else:
        print("lang_code_to_id NOT found.")
        # Try convert_tokens_to_ids
        lang_id = tokenizer.convert_tokens_to_ids("hin_Deva")
        print(f"Token 'hin_Deva' ID: {lang_id}")
        
    print("Test translation...")
    src_text = "Hello, how are you?"
    # For NLLB, setting src_lang is important
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(src_text, return_tensors="pt")
    
    forced_id = tokenizer.convert_tokens_to_ids("hin_Deva")
    
    generated_tokens = model.generate(
        **inputs, 
        forced_bos_token_id=forced_id,
        max_length=128
    )
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    print(f"Result: {result}")
    print("NLLB-200 verified successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
