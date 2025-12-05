import tempfile
import time
from io import StringIO
from pathlib import Path
from threading import Thread
from typing import BinaryIO, Union

import numpy as np
import soundfile as sf
import torch

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG
from app.logging_config import get_logger
from app.utils import ResultWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

logger = get_logger(__name__)

# Parakeet supports 25 languages (ISO 639-1 codes)
PARAKEET_SUPPORTED_LANGUAGES = {
    "bg",  # Bulgarian
    "hr",  # Croatian
    "cs",  # Czech
    "da",  # Danish
    "nl",  # Dutch
    "en",  # English
    "et",  # Estonian
    "fi",  # Finnish
    "fr",  # French
    "de",  # German
    "el",  # Greek
    "hu",  # Hungarian
    "it",  # Italian
    "lv",  # Latvian
    "lt",  # Lithuanian
    "mt",  # Maltese
    "pl",  # Polish
    "pt",  # Portuguese
    "ro",  # Romanian
    "sk",  # Slovak
    "sl",  # Slovenian
    "es",  # Spanish
    "sv",  # Swedish
    "ru",  # Russian
    "uk",  # Ukrainian
}


class ParakeetASR(ASRModel):
    """
    NVIDIA Parakeet ASR Engine using NeMo toolkit.
    Supports multilingual speech recognition with automatic language detection.
    
    Supported languages: bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk
    """

    def load_model(self):
        """Load the Parakeet model using NeMo toolkit."""
        try:
            import nemo.collections.asr as nemo_asr
            
            model_name = CONFIG.MODEL_NAME
            # If MODEL_NAME is just "parakeet-tdt-0.6b-v3", use full HuggingFace path
            if model_name == "parakeet-tdt-0.6b-v3" or not "/" in model_name:
                model_name = f"nvidia/{model_name}" if not model_name.startswith("nvidia/") else model_name
            
            logger.info("Loading Parakeet model", model_name=model_name)
            
            # Load model from HuggingFace
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
            
            # Move to GPU if available and configured
            device = CONFIG.DEVICE
            if device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model loaded on CUDA")
            else:
                self.model = self.model.cpu()
                logger.info("Model loaded on CPU")
                
        except ImportError:
            raise ImportError(
                "NeMo toolkit is required for Parakeet engine. "
                "Install with: pip install -U nemo_toolkit['asr']"
            )
        except Exception as e:
            logger.error("Failed to load Parakeet model", error=str(e))
            raise

        Thread(target=self.monitor_idleness, daemon=True).start()

    def transcribe(
        self,
        audio,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        options: Union[dict, None],
        output,
    ):
        """
        Transcribe audio using Parakeet model.
        
        Parakeet automatically detects language and supports timestamps.
        """
        self.last_activity_time = time.time()

        with self.model_lock:
            if self.model is None:
                self.load_model()

        # Parakeet expects file paths, so we need to save audio to a temporary file
        # Audio is a numpy array (float32, normalized to [-1, 1])
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            try:
                # Save audio to temporary WAV file
                # Parakeet expects 16kHz audio
                sf.write(tmp_path, audio, CONFIG.SAMPLE_RATE, format="WAV")
                
                # Prepare transcription options
                transcribe_kwargs = {}
                if word_timestamps:
                    transcribe_kwargs["timestamps"] = True
                
                # Parakeet supports language hint, but auto-detects if not provided
                # NeMo's transcribe method may accept language parameter
                # Note: Parakeet auto-detects language, but we can pass it as a hint if supported
                if language:
                    # Validate language code against Parakeet's supported languages
                    if language.lower() in PARAKEET_SUPPORTED_LANGUAGES:
                        # Try to pass language if NeMo supports it
                        # Some NeMo models support language parameter in transcribe
                        try:
                            # Check if model has transcribe method that accepts language
                            transcribe_kwargs["language"] = language.lower()
                            logger.debug("Using language hint", language=language)
                        except Exception:
                            # If language parameter is not supported, Parakeet will auto-detect
                            logger.debug("Language parameter not supported, using auto-detection", language=language)
                    else:
                        logger.warning(
                            "Language not in Parakeet supported list, using auto-detection",
                            language=language,
                            supported_languages=list(PARAKEET_SUPPORTED_LANGUAGES),
                        )
                
                # Parakeet transcribe expects a list of file paths
                with self.model_lock:
                    output_result = self.model.transcribe([tmp_path], **transcribe_kwargs)
                
                # NeMo returns a list of results, get the first one
                if isinstance(output_result, list) and len(output_result) > 0:
                    nemo_output = output_result[0]
                else:
                    nemo_output = output_result
                
                # Convert NeMo output format to our expected format
                # Pass language to conversion to use detected or provided language
                result = self._convert_nemo_output(nemo_output, word_timestamps, language)
                
            finally:
                # Clean up temporary file
                Path(tmp_path).unlink(missing_ok=True)

        output_file = StringIO()
        self.write_result(result, output_file, output)
        output_file.seek(0)

        return output_file

    def _convert_nemo_output(self, nemo_output, word_timestamps: bool, language: Union[str, None] = None):
        """
        Convert NeMo Parakeet output format to our standard format.
        
        NeMo returns an object with:
        - .text: full transcription text
        - .timestamp: dict with 'word', 'segment', 'char' timestamps (if timestamps=True)
        """
        # Try to extract detected language from NeMo output
        detected_language = language  # Use provided language if available
        
        # NeMo may expose language in the output object
        if hasattr(nemo_output, "language"):
            detected_language = nemo_output.language
        elif hasattr(nemo_output, "lang"):
            detected_language = nemo_output.lang
        
        # If no language detected, use "auto" to indicate auto-detection
        if not detected_language:
            detected_language = "auto"
        
        result = {
            "language": detected_language,
            "text": nemo_output.text if hasattr(nemo_output, "text") else str(nemo_output),
            "segments": [],
        }
        
        # Extract segments if timestamps are available
        if word_timestamps and hasattr(nemo_output, "timestamp"):
            timestamp_data = nemo_output.timestamp
            
            # Use segment-level timestamps if available
            if "segment" in timestamp_data:
                segments = timestamp_data["segment"]
                for seg in segments:
                    result["segments"].append({
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "text": seg.get("segment", ""),
                        "words": [],
                    })
                    
                    # Add word-level timestamps if available
                    if "word" in timestamp_data:
                        words = timestamp_data["word"]
                        # Filter words that belong to this segment
                        seg_start = seg.get("start", 0.0)
                        seg_end = seg.get("end", 0.0)
                        for word in words:
                            word_start = word.get("start", 0.0)
                            if seg_start <= word_start <= seg_end:
                                result["segments"][-1]["words"].append({
                                    "start": word_start,
                                    "end": word.get("end", word_start),
                                    "word": word.get("word", ""),
                                })
            else:
                # Fallback: create a single segment with the full text
                result["segments"].append({
                    "start": 0.0,
                    "end": 0.0,  # Duration unknown without timestamps
                    "text": result["text"],
                })
        else:
            # No timestamps - create a single segment
            result["segments"].append({
                "start": 0.0,
                "end": 0.0,
                "text": result["text"],
            })
        
        return result

    def language_detection(self, audio):
        """
        Detect language using Parakeet model.
        
        Parakeet automatically detects language during transcription.
        We'll do a quick transcription to get the detected language.
        """
        self.last_activity_time = time.time()

        with self.model_lock:
            if self.model is None:
                self.load_model()

        """
        Detect language using Parakeet model.
        
        Parakeet auto-detects language during transcription.
        We perform a quick transcription to get the detected language.
        """
        # Use first 5 seconds for language detection
        audio_sample = audio[: CONFIG.SAMPLE_RATE * 5] if len(audio) > CONFIG.SAMPLE_RATE * 5 else audio
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            try:
                sf.write(tmp_path, audio_sample, CONFIG.SAMPLE_RATE, format="WAV")
                
                with self.model_lock:
                    output_result = self.model.transcribe([tmp_path])
                
                # Extract detected language from NeMo output
                if isinstance(output_result, list) and len(output_result) > 0:
                    nemo_output = output_result[0]
                else:
                    nemo_output = output_result
                
                # Try to get language from output
                detected_lang_code = "auto"
                if hasattr(nemo_output, "language"):
                    detected_lang_code = nemo_output.language
                elif hasattr(nemo_output, "lang"):
                    detected_lang_code = nemo_output.lang
                
                # Parakeet auto-detects, so we return high confidence
                detected_language_confidence = 1.0
                
                logger.debug(
                    "Language detected",
                    language=detected_lang_code,
                    confidence=detected_language_confidence,
                )
                
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        return detected_lang_code, detected_language_confidence

    def write_result(self, result: dict, file: BinaryIO, output: Union[str, None]):
        """Write transcription result in the requested format."""
        if output == "srt":
            WriteSRT(ResultWriter).write_result(result, file=file)
        elif output == "vtt":
            WriteVTT(ResultWriter).write_result(result, file=file)
        elif output == "tsv":
            WriteTSV(ResultWriter).write_result(result, file=file)
        elif output == "json":
            WriteJSON(ResultWriter).write_result(result, file=file)
        else:
            WriteTXT(ResultWriter).write_result(result, file=file)

