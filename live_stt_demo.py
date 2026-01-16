import sounddevice as sd
import numpy as np
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
from huggingface_hub import hf_hub_download
import json
import os
import argparse
import sys

# Config
MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
SAMPLE_RATE = 16000
CHUNK_DURATION = 4 # seconds

def get_args():
    parser = argparse.ArgumentParser(description="Indic Conformer Live STT Demo")
    parser.add_argument("--lang", type=str, default="hi", help="Language code (e.g., hi, ta, bn, en). Default: hi")
    parser.add_argument("--list-langs", action="store_true", help="List available languages")
    return parser.parse_args()

def setup_processor(lang):
    print(f"Setting up processor for language: {lang}")
    try:
        # 1. Download assets/vocab.json
        print("Downloading vocab file...")
        vocab_path = hf_hub_download(repo_id=MODEL_ID, filename="assets/vocab.json")
        
        # 2. Extract specific language vocab
        with open(vocab_path, "r", encoding="utf-8") as f:
            all_vocabs = json.load(f)
            
        if lang not in all_vocabs:
            print(f"Error: Language '{lang}' not found in vocabulary.")
            print(f"Available languages: {list(all_vocabs.keys())}")
            sys.exit(1)
            
        vocab_list = all_vocabs[lang]
        
        # 3. Convert list to dict {token: id} and save to temp file
        vocab_dict = {token: i for i, token in enumerate(vocab_list)}
        
        local_vocab_file = f"vocab_{lang}.json"
        with open(local_vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False)
            
        # 4. Create Tokenizer
        # Notes:
        # - Vocab has | at 256, which matches BLANK_ID in config. So | is the blank/pad token.
        # - Vocab uses ▁ (U+2581) for space/start-of-word.
        tokenizer = Wav2Vec2CTCTokenizer(
            local_vocab_file, 
            unk_token="<unk>", 
            pad_token="|", 
            word_delimiter_token="|" # This might be wrong if | is blank, but we handle replacement manually
        )
        
        # 5. Create Feature Extractor
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, 
            sampling_rate=16000, 
            padding_value=0.0, 
            do_normalize=True, 
            return_attention_mask=True
        )
        
        # 6. Create Processor
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        return processor
        
    except Exception as e:
        print(f"Error setting up processor: {e}")
        sys.exit(1)

def main():
    args = get_args()
    
    # Handle --list-langs check requires downloading the file first, handled in setup but let's do quick check if possible
    # We will just proceed and fail if lang invalid
    
    print("="*50)
    print(f"Initializing Indic-Conformer STT Demo")
    print(f"Model: {MODEL_ID}")
    print(f"Language: {args.lang}")
    print("="*50)
    
    # Load Processor
    processor = setup_processor(args.lang)
    
    # Load Model
    print("Loading model (this may take a while)...")
    try:
        model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
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
            
            # Flatten audio to 1D array
            audio_input = audio_input.flatten()
            
            # Prepare inputs - SKIP processor for inputs, pass raw tensor
            # inputs = processor(audio_input, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            
            # Convert to tensor and add batch dim
            input_tensor = torch.tensor(audio_input).unsqueeze(0)
            
            # Forward pass
            with torch.no_grad():
                # Pass raw audio tensor directly
                outputs = model(input_tensor)
                
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    # Fallback if model just returns tensor
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Post-process for SentencePiece
            transcription = transcription.replace('▁', ' ').strip()
            
            # clear line
            print(" " * 50, end="\r")
            if transcription.strip():
                print(f"Transcription ({args.lang}): {transcription}")
            else:
                pass 
                # print(f"[No speech detected]") # Clean output

    except KeyboardInterrupt:
        print("\n\nStopping...")
        try:
            os.remove(f"vocab_{args.lang}.json")
        except:
            pass
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
