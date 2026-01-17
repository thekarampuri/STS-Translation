@echo off
echo [1/3] Activating environment: indic-tts...
call conda activate indic-tts

echo [2/3] Checking for NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab');"

echo [3/3] Starting Indic STT & Translate & TTS on http://127.0.0.1:8001...
python app.py
pause
