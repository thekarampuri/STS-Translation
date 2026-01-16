import json
from huggingface_hub import hf_hub_download

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
try:
    vocab_file = hf_hub_download(repo_id=MODEL_ID, filename="assets/vocab.json")
    with open(vocab_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    hi_vocab = data.get('hi', [])
    ta_vocab = data.get('ta', [])
    en_vocab = data.get('en', [])
    
    print(f"Hindi vocab size: {len(hi_vocab)}")
    print(f"Tamil vocab size: {len(ta_vocab)}")
    
    if hi_vocab == ta_vocab:
        print("Hindi and Tamil vocabs are IDENTICAL")
    else:
        print("Hindi and Tamil vocabs are DIFFERENT")
        print(f"First 5 diffs: {[(h, t) for h, t in zip(hi_vocab, ta_vocab) if h != t][:5]}")
        
    if hi_vocab == en_vocab:
         print("Hindi and English vocabs are IDENTICAL")
    else:
         print("Hindi and English vocabs are DIFFERENT")

    # Save Hindi vocab to file to see format
    with open("vocab_hi.json", "w", encoding="utf-8") as f:
        # Create dict {token: id}
        vocab_dict = {token: i for i, token in enumerate(hi_vocab)}
        json.dump(vocab_dict, f, indent=2, ensure_ascii=False)
    print("Saved vocab_hi.json")

except Exception as e:
    print(e)
