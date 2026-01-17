from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import librosa
import io
import os
import shutil
import warnings
import subprocess
import imageio_ffmpeg
import soundfile as sf
from typing import Optional

# Suppress warnings
warnings.filterwarnings("ignore")

# Force imageio-ffmpeg to be in PATH for librosa/whisper
try:
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_bin)
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + ffmpeg_dir
    print(f"FFmpeg path injected: {ffmpeg_dir}")
except Exception as e:
    print(f"Warning: Could not inject FFmpeg path: {e}")

app = FastAPI(title="Indic STT & Translate")

# Config
MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
MT_MODEL_ID = "facebook/nllb-200-distilled-600M"
SAMPLE_RATE = 16000

# Global model containers
model_stt_indic = None
model_stt_en = None
model_trans = None
tokenizer_trans = None

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

@app.on_event("startup")
async def load_models():
    global model_stt_indic, model_stt_en, model_trans, tokenizer_trans
    
    print(f"Loading Indic STT model...")
    model_stt_indic = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model_stt_indic.eval()
    
    print(f"Loading English STT model (Whisper Tiny)...")
    model_stt_en = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    
    print(f"Loading Translation model (NLLB-200 600M)...")
    tokenizer_trans = AutoTokenizer.from_pretrained(MT_MODEL_ID)
    model_trans = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL_ID)
    model_trans.eval()
    
    print("All models loaded successfully.")

def translate_text(text, src_lang, tgt_lang):
    if not text or src_lang == tgt_lang:
        return text
    
    try:
        src_code = LANG_CODES.get(src_lang)
        tgt_code = LANG_CODES.get(tgt_lang)
        
        if not src_code or not tgt_code:
            return text
            
        # NLLB-200 usage
        tokenizer_trans.src_lang = src_code
        inputs = tokenizer_trans(text, return_tensors="pt")
        
        # Get target language token ID
        tgt_id = tokenizer_trans.convert_tokens_to_ids(tgt_code)
        
        with torch.no_grad():
            generated_tokens = model_trans.generate(
                **inputs, 
                forced_bos_token_id=tgt_id,
                max_length=128
            )
        return tokenizer_trans.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
    except Exception as e:
        print(f"Translation error: {e}")
        return f"{text} (Translation failed)"

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Indic STT & Translate</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #6366f1;
                --primary-hover: #4f46e5;
                --bg: #0f172a;
                --card-bg: rgba(30, 41, 59, 0.7);
                --text: #f8fafc;
                --text-dim: #94a3b8;
            }

            body {
                margin: 0;
                padding: 0;
                font-family: 'Outfit', sans-serif;
                background: radial-gradient(circle at top left, #1e1b4b, #0f172a);
                color: var(--text);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }

            .container {
                width: 95%;
                max-width: 700px;
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 24px;
                padding: 40px;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                text-align: center;
            }

            h1 {
                font-size: 2.5rem;
                margin-bottom: 8px;
                background: linear-gradient(to right, #818cf8, #c084fc);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            p.subtitle {
                color: var(--text-dim);
                margin-bottom: 32px;
            }

            .controls {
                display: flex;
                flex-direction: column;
                gap: 20px;
                margin-bottom: 32px;
            }

            select {
                background: #1e293b;
                border: 1px solid #334155;
                color: white;
                padding: 12px 15px;
                border-radius: 12px;
                font-size: 0.95rem;
                outline: none;
                cursor: pointer;
            }

            .btn-container {
                display: flex;
                justify-content: center;
                margin-top: 10px;
            }

            button {
                padding: 14px 32px;
                border-radius: 12px;
                font-weight: 600;
                cursor: pointer;
                border: none;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            #recordBtn { background: var(--primary); color: white; }
            #recordBtn.recording { background: #ef4444; animation: pulse 1.5s infinite; }

            @keyframes pulse {
                0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
                70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
                100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
            }

            .result-area {
                margin-top: 20px;
                text-align: left;
                background: rgba(15, 23, 42, 0.5);
                border-radius: 16px;
                padding: 24px;
                min-height: 80px;
                border: 1px solid rgba(255, 255, 255, 0.05);
            }

            .result-label {
                font-size: 0.75rem;
                color: var(--text-dim);
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 10px;
                display: block;
            }

            #transcription, #translationText {
                font-size: 1.1rem;
                line-height: 1.5;
                white-space: pre-wrap;
            }

            .status { margin-top: 15px; font-size: 0.85rem; color: var(--text-dim); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Indic STT & Translate</h1>
            <p class="subtitle">Real-time voice transcription and translation</p>
            
            <div class="controls">
                <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 200px; text-align: left;">
                        <span class="result-label">Source Language</span>
                        <select id="langSelect" style="width: 100%;">
                            <option value="hi">Hindi (हिन्दी)</option>
                            <option value="en">English</option>
                            <option value="mr">Marathi (मराठी)</option>
                            <option value="te">Telugu (తెలుగు)</option>
                            <option value="ta">Tamil (தமிழ்)</option>
                            <option value="ml">Malayalam (മലയാളം)</option>
                            <option value="kn">Kannada (ಕನ್ನಡ)</option>
                            <option value="gu">Gujarati (ગુજરાતી)</option>
                            <option value="bn">Bengali (বাংলা)</option>
                            <option value="pa">Punjabi (ਪੰਜਾਬੀ)</option>
                        </select>
                    </div>
                    <div style="flex: 1; min-width: 200px; text-align: left;">
                        <span class="result-label">Target Language</span>
                        <select id="targetSelect" style="width: 100%;">
                            <option value="">STT Only (No Translation)</option>
                            <option value="en">English</option>
                            <option value="hi">Hindi (हिन्दी)</option>
                            <option value="mr">Marathi (मराठी)</option>
                            <option value="te">Telugu (తెలుగు)</option>
                            <option value="ta">Tamil (தமிழ்)</option>
                            <option value="ml">Malayalam (മലയാളം)</option>
                            <option value="kn">Kannada (ಕನ್ನಡ)</option>
                            <option value="gu">Gujarati (ગુજરાતી)</option>
                            <option value="bn">Bengali (বাংলা)</option>
                        </select>
                    </div>
                </div>

                <div class="btn-container">
                    <button id="recordBtn">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>
                        <span id="btnText">Start Recording</span>
                    </button>
                </div>
            </div>

            <div class="result-area">
                <span class="result-label">Transcription</span>
                <div id="transcription">Speech will appear here...</div>
            </div>

            <div id="translationArea" class="result-area" style="display: none; border-left: 4px solid var(--primary);">
                <span class="result-label">Translation</span>
                <div id="translationText"></div>
            </div>
            
            <div id="status" class="status">Ready</div>
        </div>

        <script>
            let mediaRecorder;
            let audioChunks = [];
            let stream;
            
            const recordBtn = document.getElementById('recordBtn');
            const btnText = document.getElementById('btnText');
            const langSelect = document.getElementById('langSelect');
            const targetSelect = document.getElementById('targetSelect');
            const transcriptionArea = document.getElementById('transcription');
            const translationArea = document.getElementById('translationArea');
            const translationText = document.getElementById('translationText');
            const statusArea = document.getElementById('status');

            recordBtn.addEventListener('click', async () => {
                if (mediaRecorder && mediaRecorder.state === "recording") {
                    stopRecording();
                    return;
                }

                try {
                    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];

                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            audioChunks.push(event.data);
                            sendCurrentAudio();
                        }
                    };

                    mediaRecorder.start(2000); 
                    
                    recordBtn.classList.add('recording');
                    btnText.innerText = "Stop Recording";
                    transcriptionArea.innerText = "Listening...";
                    translationArea.style.display = "none";
                    statusArea.innerText = "Recording...";
                } catch (err) {
                    console.error(err);
                    statusArea.innerText = "Mic Error: Check permissions.";
                }
            });

            function stopRecording() {
                if (mediaRecorder) mediaRecorder.stop();
                if (stream) stream.getTracks().forEach(t => t.stop());
                recordBtn.classList.remove('recording');
                btnText.innerText = "Start Recording";
                statusArea.innerText = "Done";
            }

            async function sendCurrentAudio() {
                if (audioChunks.length === 0) return;
                
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const formData = new FormData();
                formData.append('audio', audioBlob);
                formData.append('lang', langSelect.value);
                formData.append('target_lang', targetSelect.value);

                try {
                    const response = await fetch('/transcribe', { method: 'POST', body: formData });
                    const data = await response.json();
                    
                    if (data.text) {
                        transcriptionArea.innerText = data.text;
                        if (data.translated_text) {
                            translationArea.style.display = "block";
                            translationText.innerText = data.translated_text;
                        }
                        statusArea.innerText = "Live Update...";
                    }
                } catch (err) {
                    console.error(err);
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), lang: str = Form(...), target_lang: Optional[str] = Form(None)):
    temp_filename = f"temp_{os.getpid()}.webm"
    wav_filename = f"temp_{os.getpid()}.wav"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([
            ffmpeg_path, "-y", "-i", temp_filename, 
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", wav_filename
        ], check=True, capture_output=True)
        # Load the converted wav file using soundfile (no FFmpeg required for WAV)
        y, sr = sf.read(wav_filename)
        if len(y) == 0: return {"text": ""}

        if lang == "en":
            # Pass numpy array to Whisper to avoid filename-based FFmpeg triggers
            res = model_stt_en(y)
            text = res["text"].strip()
        else:
            # Use Indic-Conformer for Indian languages
            input_tensor = torch.from_numpy(y).float().unsqueeze(0)
            with torch.no_grad():
                transcription = model_stt_indic(input_tensor, lang=lang)
            text = transcription.replace('▁', ' ').strip() if isinstance(transcription, str) else ""
        
        translated_text = ""
        if text and target_lang and target_lang != lang:
            translated_text = translate_text(text, lang, target_lang)
            
        return {"text": text, "translated_text": translated_text}

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
    finally:
        for f in [temp_filename, wav_filename]:
            if os.path.exists(f): 
                try: os.remove(f)
                except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
