import torch
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from config.settings import settings
from core.device_manager import device_manager
from core.model_manager import model_manager

logger = logging.getLogger(__name__)

LANG_CODES = {
    "hi": "hin_Deva", "mr": "mar_Deva", "te": "tel_Telu", "ta": "tam_Taml",
    "ml": "mal_Mlym", "kn": "kan_Knda", "gu": "guj_Gujr", "bn": "ben_Beng",
    "as": "asm_Beng", "pa": "pan_Guru", "en": "eng_Latn"
}

class MTEngine:
    def __init__(self):
        self.tokenizer = None
        
    def load_model(self):
        logger.info(f"Loading Translation model from {settings.MT_MODEL_PATH}...")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MT_MODEL_PATH)
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.MT_MODEL_PATH,
            trust_remote_code=True,
            use_safetensors=True,
            low_cpu_mem_usage=False
        )
        if device_manager.is_cuda():
            model = model.to("cuda")
        model.eval()
        return model

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not text or src_lang == tgt_lang:
            return text
            
        src_code = LANG_CODES.get(src_lang)
        tgt_code = LANG_CODES.get(tgt_lang)
        
        if not src_code or not tgt_code:
            logger.warning(f"Unsupported language pair: {src_lang} -> {tgt_lang}")
            return text

        try:
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
            return f"{text} (Translation failed)"

mt_engine = MTEngine()
