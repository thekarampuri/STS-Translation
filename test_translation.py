import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Language code mapping for NLLB-200 (FLORES-200)
LANG_CODES = {
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "ml": "mal_Mlym",
    "kn": "kan_Knda",
    "gu": "guj_Gujr",
    "bn": "ben_Beng",
    "as": "asm_Beng",
    "pa": "pan_Guru",
    "en": "eng_Latn"
}

MT_MODEL_ID = "./nllb-safe" # Use local converted model

print("Loading NLLB model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MT_MODEL_ID)
    print("Tokenizer loaded")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MT_MODEL_ID, 
        trust_remote_code=True,
        use_safetensors=True,
        low_cpu_mem_usage=False
    )
    print(f"Initial model device: {model.device}")
    if torch.cuda.is_available():
        model = model.to("cuda")
    print(f"Model loaded on {model.device}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

model.eval()

def test_translate(text, src_lang, tgt_lang):
    print(f"\n{'='*60}")
    print(f"Testing: {src_lang} -> {tgt_lang}")
    print(f"Input: {text}")
    
    src_code = LANG_CODES.get(src_lang)
    tgt_code = LANG_CODES.get(tgt_lang)
    
    print(f"Source code: {src_code}")
    print(f"Target code: {tgt_code}")
    
    # Method 1: Using src_lang attribute
    tokenizer.src_lang = src_code
    inputs = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_code)
    print(f"Target token ID: {tgt_id}")
    
    with torch.inference_mode():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tgt_id,
            max_length=128,
            num_beams=1,
            do_sample=False
        )
    
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    print(f"Output: {result}")
    
    # Method 2: Using tgt_lang in tokenizer call
    print("\nAlternative method:")
    inputs2 = tokenizer(text, return_tensors="pt", src_lang=src_code)
    if torch.cuda.is_available():
        inputs2 = {k: v.to("cuda") for k, v in inputs2.items()}
    
    with torch.inference_mode():
        generated_tokens2 = model.generate(
            **inputs2,
            forced_bos_token_id=tgt_id,
            max_length=128,
            num_beams=5,
            do_sample=False
        )
    
    result2 = tokenizer.batch_decode(generated_tokens2, skip_special_tokens=True)[0]
    print(f"Output (alt): {result2}")

# Test cases
test_translate("नमस्ते, आप कैसे हैं?", "hi", "en")
test_translate("Hello, how are you?", "en", "hi")
test_translate("नमस्ते, आप कैसे हैं?", "hi", "mr")
