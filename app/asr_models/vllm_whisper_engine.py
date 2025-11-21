import time
from io import StringIO
from threading import Thread
from typing import BinaryIO, Union
import tempfile
import os

from openai import OpenAI

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG
from app.logging_config import get_logger
from app.utils import ResultWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

logger = get_logger(__name__)


class VLLMWhisperASR(ASRModel):
    """
    ASR engine using vLLM to serve Whisper models.
    vLLM provides high-performance inference for Whisper models with optimizations
    like continuous batching and PagedAttention.
    """

    def __init__(self):
        super().__init__()
        self.client = None
        self.vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.vllm_api_key = os.getenv("VLLM_API_KEY", "dummy-key")
        # vLLM expects full HuggingFace model paths (e.g., "openai/whisper-large-v3")
        # If VLLM_MODEL is not set, try to construct from ASR_MODEL
        vllm_model_env = os.getenv("VLLM_MODEL")
        if vllm_model_env:
            self.vllm_model = vllm_model_env
        else:
            # Convert short model names to HuggingFace paths
            model_name = CONFIG.MODEL_NAME
            if "/" not in model_name:
                # Assume OpenAI Whisper models
                self.vllm_model = f"openai/whisper-{model_name}"
            else:
                self.vllm_model = model_name

    def load_model(self):
        """
        Initialize the vLLM client.
        Note: The model is served by vLLM, so we just need to set up the client.
        """
        self.client = OpenAI(
            base_url=self.vllm_base_url,
            api_key=self.vllm_api_key,
        )
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
            if self.client is None:
                self.load_model()

        # Save audio to temporary file for vLLM API
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio)
            tmp_file_path = tmp_file.name

        try:
            # Prepare transcription parameters
            transcription_params = {}
            if language:
                transcription_params["language"] = language
            if initial_prompt:
                transcription_params["prompt"] = initial_prompt
            if word_timestamps:
                # vLLM Whisper supports word_timestamps via response_format
                transcription_params["response_format"] = "verbose_json"

            # Call vLLM transcription API
            with open(tmp_file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model=self.vllm_model,
                    file=audio_file,
                    **transcription_params
                )

            # Convert vLLM response to our format
            if isinstance(transcription, dict):
                # verbose_json format with segments
                text = transcription.get("text", "")
                segments_data = transcription.get("segments", [])
                segments = [
                    {
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "text": seg.get("text", ""),
                    }
                    for seg in segments_data
                ]
                detected_language = transcription.get("language", language or "unknown")
            else:
                # Simple text response
                text = transcription.text if hasattr(transcription, "text") else str(transcription)
                segments = [{"start": 0.0, "end": 0.0, "text": text}]
                detected_language = language or "unknown"

            result = {
                "language": detected_language,
                "text": text,
                "segments": segments,
            }

            output_file = StringIO()
            self.write_result(result, output_file, output)
            output_file.seek(0)

            return output_file

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    def language_detection(self, audio):
        """
        Perform language detection using vLLM Whisper.
        """
        self.last_activity_time = time.time()

        with self.model_lock:
            if self.client is None:
                self.load_model()

        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio)
            tmp_file_path = tmp_file.name

        try:
            # Use verbose_json to get language info
            with open(tmp_file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model=self.vllm_model,
                    file=audio_file,
                    response_format="verbose_json",
                )

            if isinstance(transcription, dict):
                detected_lang_code = transcription.get("language", "unknown")
                # vLLM doesn't always provide confidence, so we use 1.0 as default
                detected_language_confidence = transcription.get("language_probability", 1.0)
            else:
                detected_lang_code = "unknown"
                detected_language_confidence = 1.0

            return detected_lang_code, detected_language_confidence

        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    def release_model(self):
        """
        Release the vLLM client.
        Note: The actual model is managed by vLLM server, so we just clear the client reference.
        """
        self.client = None
        logger.info("vLLM client released due to timeout", timeout=CONFIG.MODEL_IDLE_TIMEOUT)

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

