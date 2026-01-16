from huggingface_hub import list_repo_files
import sys

try:
    files = list_repo_files("ai4bharat/indic-conformer-600m-multilingual")
    for f in files:
        print(f)
except Exception as e:
    print(e)
