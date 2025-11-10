import gc
import time
from dataclasses import dataclass
from io import StringIO
from threading import Thread
from typing import BinaryIO, Union

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG
from app.utils import ResultWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT


@dataclass
class Segment:
    start: float
    end: float
    text: str


class VoxtralASR(ASRModel):
    def __init__(self):
        super().__init__()
        self.processor = None

    def load_model(self):
        # Handle model name - support both full HuggingFace paths and short names
        model_name = CONFIG.MODEL_NAME
        if "/" not in model_name:
            # If no organization prefix, assume it's a Voxtral model
            model_name = f"mistralai/{model_name}"

        device = CONFIG.DEVICE
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        torch_dtype = torch.float32
        if CONFIG.MODEL_QUANTIZATION == "float16":
            torch_dtype = torch.float16
        elif CONFIG.MODEL_QUANTIZATION == "int8":
            torch_dtype = torch.int8

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device if device == "cuda" else None,
        )

        if device == "cpu":
            self.model = self.model.to(device)

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
        self.last_activity_time = time.time()

        with self.model_lock:
            if self.model is None:
                self.load_model()

        # Process audio - Voxtral processor expects numpy array directly
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=CONFIG.SAMPLE_RATE)

        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate transcription
        generate_kwargs = {}
        if language:
            # Voxtral uses language codes, but we need to check if it supports forced language
            # For now, we'll let the model detect/use the language naturally
            pass
        if initial_prompt:
            # Voxtral may support prompt, but need to check the API
            pass

        with self.model_lock:
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generate_kwargs)

        # Decode the transcription
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Convert to the expected format with segments
        # Since Voxtral doesn't provide timestamps by default, we'll create a single segment
        # covering the entire audio duration
        audio_duration = len(audio) / CONFIG.SAMPLE_RATE
        result = {
            "language": language or "unknown",
            "text": transcription,
            "segments": [
                Segment(
                    start=0.0,
                    end=audio_duration,
                    text=transcription,
                )
            ],
        }

        output_file = StringIO()
        self.write_result(result, output_file, output)
        output_file.seek(0)

        return output_file

    def language_detection(self, audio):
        self.last_activity_time = time.time()

        with self.model_lock:
            if self.model is None:
                self.load_model()

        # Voxtral can detect language during transcription
        # We'll do a short transcription to detect the language
        inputs = self.processor(audio, return_tensors="pt", sampling_rate=CONFIG.SAMPLE_RATE)

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with self.model_lock:
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs)

        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Voxtral doesn't explicitly return language detection, so we'll return "unknown"
        # In a real implementation, you might want to use a separate language detection model
        # or check if Voxtral provides language info in the model outputs
        detected_lang_code = "unknown"
        detected_language_confidence = 1.0

        return detected_lang_code, detected_language_confidence

    def release_model(self):
        """
        Unloads the model and processor from memory and clears any cached GPU memory.
        """
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        torch.cuda.empty_cache()
        gc.collect()
        self.model = None
        self.processor = None
        print("Model unloaded due to timeout")

    def write_result(self, result: dict, file: BinaryIO, output: Union[str, None]):
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

