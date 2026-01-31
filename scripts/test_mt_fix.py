import sys
import os
import asyncio

# Add project root to path
sys.path.append(os.getcwd())

from pipeline.mt_engine import mt_engine
from core.model_manager import model_manager

async def test_mt():
    print("--- Starting MT Fix Verification ---")
    
    # 1. Test Loading
    print("1. Testing Model Load (Should happen once)...")
    try:
        # Initial load
        model = model_manager.load_model("mt_model", mt_engine.load_model)
        print("   Success: Model loaded.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"   FAILURE: Loading failed: {e}")
        return

    # 2. Test Re-load (Should use cache)
    print("2. Testing Cache (Should not reload)...")
    try:
        model2 = model_manager.load_model("mt_model", mt_engine.load_model)
        if model is model2:
            print("   Success: Cache working (Same object returned).")
        else:
            print("   FAILURE: Objects differ!")
    except Exception as e:
        print(f"   FAILURE: Cache test error: {e}")

    # 3. Test Translation
    print("3. Testing Translation (en -> hi)...")
    text = "Hello world"
    try:
        res = mt_engine.translate(text, "en", "hi")
        print(f"   Input: {text}")
        print(f"   Output: {res}")
        
        if "Namaste" in res or "Hello" not in res: # Rough check
             print("   Success: Translation appears to work.")
        else:
             print("   Warning: Output might be raw/garbage, checks needed.")
             
    except Exception as e:
        print(f"   FAILURE: Translation raised error: {e}")

    print("--- Verification Complete ---")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_mt())
