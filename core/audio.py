import os
import subprocess
import imageio_ffmpeg
import soundfile as sf
import logging
import tempfile
import numpy as np
import time

logger = logging.getLogger(__name__)

def inject_ffmpeg_path():
    try:
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_bin)
        if ffmpeg_dir not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + ffmpeg_dir
    except Exception as e:
        logger.warning(f"Could not inject FFmpeg path: {e}")

def probe_file(file_path):
    """
    Returns generic info string if valid, else raises Exception
    """
    # Just a basic check if ffmpeg can detect a stream
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    if not os.path.exists(file_path):
         raise FileNotFoundError(f"File {file_path} not found")
         
    # Using ffmpeg -i verify
    cmd = [ffmpeg_path, "-v", "error", "-i", file_path, "-f", "null", "-"]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def convert_webm_to_wav(webm_bytes: bytes, sample_rate: int = 16000, retries=2) -> np.ndarray:
    if not webm_bytes or len(webm_bytes) < 100:
        logger.error("Received bytes too small to be valid audio")
        return np.array([])
        
    inject_ffmpeg_path()
    
    # Write temp file
    temp_webm = None
    wav_filename = None
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(webm_bytes)
            f.flush()
            temp_webm_path = f.name
            
        wav_filename = temp_webm_path.replace(".webm", ".wav")
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        
        # Verify input
        try:
            probe_file(temp_webm_path)
        except Exception as e:
            logger.error(f"Input file validation failed: {e}")
            return np.array([])
            
        # Conversion command with explicit pcm_s16le
        cmd = [
            ffmpeg_path, "-y", 
            "-v", "error", 
            "-i", temp_webm_path, 
            "-ar", str(sample_rate), 
            "-ac", "1", 
            "-c:a", "pcm_s16le", 
            "-f", "wav", 
            wav_filename
        ]
        
        for attempt in range(retries):
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                break
            except subprocess.CalledProcessError as e:
                if attempt == retries - 1:
                    logger.error(f"FFmpeg failed: {e.stderr.decode() if e.stderr else str(e)}")
                    raise e
                time.sleep(0.1)

        # Read back
        y, sr = sf.read(wav_filename)
        return y
        
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return np.array([])
        
    finally:
        # Cleanup
        for p in [temp_webm_path, wav_filename]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass
