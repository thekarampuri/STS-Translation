import json
from huggingface_hub import hf_hub_download

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

def check_langs():
    vocab_path = hf_hub_download(repo_id=MODEL_ID, filename="assets/vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        all_vocabs = json.load(f)
    print("Available languages in vocab.json:")
    print(list(all_vocabs.keys()))

if __name__ == "__main__":
    check_langs()
