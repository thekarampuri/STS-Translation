import torch
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from config.settings import settings
from core.device_manager import device_manager
from core.model_manager import model_manager

logger = logging.getLogger(__name__)


# 2. Load NLLB safely (NO meta tensors) + 4. Correct language routing
NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "pa": "pan_Guru"
}

class MTEngine:
    def __init__(self):
        self.tokenizer = None
        
    def load_model(self):
        logger.info(f"Loading Translation model from {settings.MT_MODEL_PATH}...")
        
        # 3. Fix tokenizer regex (MANDATORY) - NOTE: Removed fix_mistral_regex=True as it crashes NLLB tokenizer with TypeError
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.MT_MODEL_PATH
        )
        
        # 1. Use the correct model class + 2. Load NLLB safely
        model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.MT_MODEL_PATH,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False
        )
        
        # 6. GPU handling (safe)
        if device_manager.is_cuda():
            model = model.to("cuda")
            
        model.eval()
        return model

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not text or src_lang == tgt_lang:
            return text
            
        src_code = NLLB_LANG_MAP.get(src_lang)
        tgt_code = NLLB_LANG_MAP.get(tgt_lang)
        
        if not src_code or not tgt_code:
            logger.warning(f"Unsupported language pair: {src_lang} -> {tgt_lang}")
            return f"{text} (Unsupported Language)"

        try:
            # 5. Fix ModelManager behavior (Load once, fail fast)
            model = model_manager.load_model("mt_model", self.load_model)
            
            self.tokenizer.src_lang = src_code
            inputs = self.tokenizer(text, return_tensors="pt")
            
            if device_manager.is_cuda():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            tgt_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
            
            with torch.inference_mode():
                generated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_id,
                    max_length=128,
                    num_beams=1,
                    do_sample=False
                )
            return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        except Exception as e:
            logger.error(f"Translation Error: {e}")
            raise e  # Fail fast

mt_engine = MTEngine()
