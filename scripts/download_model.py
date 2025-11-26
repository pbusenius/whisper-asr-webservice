#!/usr/bin/env python3
"""
Script to download Whisper models during Docker build.
Checks if models already exist before downloading to avoid redundant downloads.
"""
import os
import sys
from pathlib import Path

# Set default model if not provided
MODEL_NAME = os.getenv("ASR_MODEL", "base")
ASR_ENGINE = os.getenv("ASR_ENGINE", "faster_whisper")
MODEL_PATH = os.getenv("ASR_MODEL_PATH", "/app/models")

print(f"Checking model: {MODEL_NAME} for engine: {ASR_ENGINE}")
print(f"Model path: {MODEL_PATH}")

def check_faster_whisper_model(model_name: str, model_path: str) -> bool:
    """Check if faster-whisper model exists."""
    # faster-whisper stores models in {download_root}/faster-whisper/{model_name}/
    model_dir = Path(model_path) / "faster-whisper" / model_name
    # Check for common model files (config.json, model.bin, etc.)
    if model_dir.exists():
        # Check if directory has model files
        model_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.safetensors"))
        if model_files:
            print(f"✅ Model {model_name} already exists at {model_dir}")
            return True
    return False

def check_openai_whisper_model(model_name: str, model_path: str) -> bool:
    """Check if OpenAI Whisper model exists."""
    # OpenAI Whisper stores models as {model_name}.pt.gz
    model_file = Path(model_path) / "whisper" / f"{model_name}.pt.gz"
    if model_file.exists():
        print(f"✅ Model {model_name} already exists at {model_file}")
        return True
    # Also check for .pt files (without .gz)
    model_file_pt = Path(model_path) / "whisper" / f"{model_name}.pt"
    if model_file_pt.exists():
        print(f"✅ Model {model_name} already exists at {model_file_pt}")
        return True
    return False

def check_whisperx_model(model_name: str) -> bool:
    """Check if WhisperX model exists."""
    # WhisperX uses HuggingFace cache, check default cache location
    cache_dir = Path.home() / ".cache" / "whisperx"
    # WhisperX models are stored in HuggingFace format
    # This is a simplified check - WhisperX may download from HF at runtime
    print(f"⚠️  WhisperX models are typically downloaded from HuggingFace at runtime")
    return False

try:
    if ASR_ENGINE == "faster_whisper":
        if check_faster_whisper_model(MODEL_NAME, MODEL_PATH):
            print(f"⏭️  Skipping download - model already exists")
            sys.exit(0)
        
        from faster_whisper import WhisperModel
        print(f"Downloading faster-whisper model: {MODEL_NAME}")
        model = WhisperModel(
            model_size_or_path=MODEL_NAME,
            device="cpu",  # Use CPU during build
            compute_type="int8",
            download_root=MODEL_PATH
        )
        print(f"✅ Model {MODEL_NAME} downloaded successfully to {MODEL_PATH}")
        
    elif ASR_ENGINE == "openai_whisper":
        if check_openai_whisper_model(MODEL_NAME, MODEL_PATH):
            print(f"⏭️  Skipping download - model already exists")
            sys.exit(0)
        
        import whisper
        print(f"Downloading OpenAI Whisper model: {MODEL_NAME}")
        model = whisper.load_model(name=MODEL_NAME, download_root=MODEL_PATH)
        print(f"✅ Model {MODEL_NAME} downloaded successfully to {MODEL_PATH}")
        
    elif ASR_ENGINE == "whisperx":
        # WhisperX models are typically downloaded from HuggingFace at runtime
        # We can't easily check for them here, so we'll skip
        print(f"⚠️  WhisperX models are downloaded from HuggingFace at runtime - skipping build-time download")
        sys.exit(0)
        
    else:
        print(f"⚠️  Engine {ASR_ENGINE} - skipping model download (may download at runtime)")
        sys.exit(0)
        
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    sys.exit(1)

