import os
import subprocess
import imageio_ffmpeg
import soundfile as sf
import logging
import tempfile
import numpy as np

logger = logging.getLogger(__name__)

def inject_ffmpeg_path():
    try:
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_bin)
        if ffmpeg_dir not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + ffmpeg_dir
        logger.info(f"FFmpeg path injected: {ffmpeg_dir}")
    except Exception as e:
        logger.warning(f"Could not inject FFmpeg path: {e}")

def convert_webm_to_wav(webm_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    try:
        inject_ffmpeg_path()
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_webm:
            temp_webm.write(webm_bytes)
            temp_webm_path = temp_webm.name
        
        wav_filename = temp_webm_path.replace(".webm", ".wav")
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        
        subprocess.run([
            ffmpeg_path, "-y", "-loglevel", "quiet", "-i", temp_webm_path, 
            "-ar", str(sample_rate), "-ac", "1", "-f", "wav", wav_filename
        ], check=True)
        
        y, sr = sf.read(wav_filename)
        
        # Cleanup
        try:
            os.remove(temp_webm_path)
            os.remove(wav_filename)
        except:
            pass
            
        return y
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return np.array([])
