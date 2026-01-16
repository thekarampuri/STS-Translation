import sounddevice as sd
import numpy as np
import torch
from transformers import AutoModel
from huggingface_hub import hf_hub_download
import json
import os
import argparse
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Config
MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
SAMPLE_RATE = 16000
CHUNK_DURATION = 4 # seconds

def get_args():
    parser = argparse.ArgumentParser(description="Indic Conformer Live STT Demo")
    parser.add_argument("--lang", type=str, default="hi", help="Language code (e.g., hi, ta, bn, en). Default: hi")
    return parser.parse_args()

def validate_lang(lang):
    """Downloads vocab only to validate language support"""
    try:
        vocab_path = hf_hub_download(repo_id=MODEL_ID, filename="assets/vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            all_vocabs = json.load(f)
        if lang not in all_vocabs:
            print(f"Error: Language '{lang}' not found.")
            print(f"Supported languages: {list(all_vocabs.keys())}")
            sys.exit(1)
        return True
    except Exception as e:
        print(f"Warning: Could not validate language support: {e}")
        return False

def main():
    args = get_args()
    
    print("="*50)
    print(f"Initializing Indic-Conformer STT Demo")
    print(f"Model: {MODEL_ID}")
    print(f"Language: {args.lang}")
    print("="*50)
    
    # Validate language
    validate_lang(args.lang)
    
    # Load Model
    print("Loading model (this may take a while)...")
    try:
        # Note: AutoModelForCTC fails with this custom model architecture.
        # We use AutoModel and let it load the custom code from the repo.
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"\nStarting recording loop.")
    print(f"Format: Mono, {SAMPLE_RATE}Hz")
    print(f"Chunk Duration: {CHUNK_DURATION} seconds")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            print(f"Listening ({args.lang})...", end="\r", flush=True)
            
            # Record audio
            audio_input = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait() # Wait until recording is finished
            
            print("Processing...      ", end="\r", flush=True)
            
            # Convert to tensor and add batch dim
            # This model expects a raw audio tensor of shape (batch, time)
            input_tensor = torch.from_numpy(audio_input.flatten()).unsqueeze(0)
            
            # Forward pass
            with torch.no_grad():
                # The custom forward() method in Indic-Conformer handles CTC decoding internally
                # It requires the 'lang' parameter to select the correct output head.
                transcription = model(input_tensor, lang=args.lang)
            
            # Post-process for SentencePiece (model returns string with ▁ for spaces)
            if isinstance(transcription, str):
                transcription = transcription.replace('▁', ' ').strip()
            else:
                transcription = ""
            
            # Print result
            print(" " * 50, end="\r")
            if transcription:
                print(f"Transcription ({args.lang}): {transcription}")

    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
