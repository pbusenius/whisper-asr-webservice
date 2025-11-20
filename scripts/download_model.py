#!/usr/bin/env python3
"""
Script to download Whisper models during Docker build.
"""
import os
import sys

# Set default model if not provided
MODEL_NAME = os.getenv("ASR_MODEL", "base")
ASR_ENGINE = os.getenv("ASR_ENGINE", "faster_whisper")
MODEL_PATH = os.getenv("ASR_MODEL_PATH", "/app/models")

print(f"Downloading model: {MODEL_NAME} for engine: {ASR_ENGINE}")
print(f"Model path: {MODEL_PATH}")

try:
    if ASR_ENGINE == "faster_whisper":
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
        import whisper
        print(f"Downloading OpenAI Whisper model: {MODEL_NAME}")
        model = whisper.load_model(name=MODEL_NAME, download_root=MODEL_PATH)
        print(f"✅ Model {MODEL_NAME} downloaded successfully to {MODEL_PATH}")
        
    elif ASR_ENGINE == "whisperx":
        import whisperx
        print(f"Downloading WhisperX model: {MODEL_NAME}")
        model = whisperx.load_model(
            MODEL_NAME,
            device="cpu",
            compute_type="int8"
        )
        print(f"✅ Model {MODEL_NAME} downloaded successfully")
        
    else:
        print(f"⚠️  Engine {ASR_ENGINE} - skipping model download (may download at runtime)")
        sys.exit(0)
        
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    sys.exit(1)

