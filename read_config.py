from huggingface_hub import hf_hub_download
import json

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
try:
    config_file = hf_hub_download(repo_id=MODEL_ID, filename="config.json")
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(json.dumps(config, indent=2))
except Exception as e:
    print(e)
