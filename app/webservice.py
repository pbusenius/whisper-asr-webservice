import importlib.metadata
import os
from os import path
from typing import Annotated, Optional, Union
from urllib.parse import quote

import click
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from whisper import tokenizer

from app.config import CONFIG
from app.factory.asr_model_factory import ASRModelFactory
from app.logging_config import setup_structlog, get_logger
from app.middleware.structured_logging import StructuredLoggingMiddleware
from app.telemetry import get_metrics_reader, setup_telemetry
from app.utils import load_audio

# Set up structured logging
setup_structlog(
    service_name=CONFIG.OTEL_SERVICE_NAME,
    log_level=CONFIG.LOG_LEVEL,
    use_json=CONFIG.LOG_JSON,
    use_colors=CONFIG.LOG_COLORS,
)

logger = get_logger(__name__)

# Log warning if HF_TOKEN is missing for WhisperX
if CONFIG.ASR_ENGINE == "whisperx" and CONFIG.HF_TOKEN == "":
    logger.warning(
        "HF_TOKEN not set",
        message="You must set the HF_TOKEN environment variable to download the diarization model used by WhisperX",
        asr_engine=CONFIG.ASR_ENGINE,
    )

asr_model = ASRModelFactory.create_asr_model()
asr_model.load_model()

LANGUAGE_CODES = sorted(tokenizer.LANGUAGES.keys())

projectMetadata = importlib.metadata.metadata("asr-api")

app = FastAPI(
    title="ASR-API",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

# Add structured logging middleware
app.add_middleware(StructuredLoggingMiddleware)

# Set up OpenTelemetry instrumentation
setup_telemetry(app, service_name=CONFIG.OTEL_SERVICE_NAME)

assets_path = os.getcwd() + "/swagger-ui-assets"
if path.exists(assets_path + "/swagger-ui.css") and path.exists(assets_path + "/swagger-ui-bundle.js"):
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")

    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )

    applications.get_swagger_ui_html = swagger_monkey_patch


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.get("/health", tags=["Monitoring"], status_code=200, summary="Health check", description="Health check endpoint for monitoring and load balancers")
async def health():
    """
    Health check endpoint.
    
    Returns a simple status object indicating the service is healthy.
    Used by monitoring systems, load balancers, and Kubernetes readiness probes.
    
    Returns:
        dict: Status object with "status": "ok"
    """
    return {"status": "ok"}


@app.get("/debug/cuda", tags=["Debug"], status_code=200, summary="CUDA debug info", description="Debug endpoint to check CUDA availability and device info")
async def debug_cuda():
    """
    Debug endpoint to check CUDA availability.
    
    Returns information about CUDA availability, PyTorch CUDA support, and device information.
    """
    import torch
    import sys
    
    debug_info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "pytorch_cuda_available": torch.cuda.is_available(),
        "pytorch_cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "pytorch_cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config_device": CONFIG.DEVICE,
        "config_quantization": CONFIG.MODEL_QUANTIZATION,
        "asr_engine": CONFIG.ASR_ENGINE,
    }
    
    # Check faster-whisper / CTranslate2 CUDA support
    try:
        import ctranslate2
        debug_info["ctranslate2_cuda_available"] = ctranslate2.get_cuda_device_count() > 0
        debug_info["ctranslate2_cuda_device_count"] = ctranslate2.get_cuda_device_count()
    except Exception as e:
        debug_info["ctranslate2_error"] = str(e)
    
    # Check if model is loaded and what device it's using
    try:
        if hasattr(asr_model, 'model') and asr_model.model is not None:
            if hasattr(asr_model.model, 'device'):
                debug_info["model_device"] = str(asr_model.model.device)
            elif hasattr(asr_model.model, '_model'):
                # faster-whisper uses _model attribute
                debug_info["model_loaded"] = True
                debug_info["model_type"] = type(asr_model.model).__name__
    except Exception as e:
        debug_info["model_check_error"] = str(e)
    
    return debug_info


@app.get(CONFIG.OTEL_PROMETHEUS_METRICS_PATH, tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint for OpenTelemetry metrics.
    
    Returns Prometheus-formatted metrics that can be scraped by monitoring systems.
    """
    metrics_reader = get_metrics_reader()
    if metrics_reader is None:
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse("OpenTelemetry metrics not enabled", status_code=503)

    # Get the Prometheus registry from the metric reader
    # PrometheusMetricReader exposes metrics via a Prometheus registry
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    # Access the registry from the PrometheusMetricReader
    # The registry is stored in the _collector attribute
    try:
        registry = metrics_reader._collector._registry
    except AttributeError:
        # Try alternative access pattern
        try:
            registry = metrics_reader._registry
        except AttributeError:
            # Fallback to default registry
            from prometheus_client import REGISTRY

            registry = REGISTRY

    return Response(content=generate_latest(registry=registry), media_type=CONTENT_TYPE_LATEST)


@app.post("/asr", tags=["Endpoints"])
async def asr(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    vad_filter: Annotated[
        bool | None,
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
        ),
    ] = False,
    word_timestamps: bool = Query(
        default=False,
        description="Word level timestamps",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
    ),
    diarize: bool = Query(
        default=False,
        description="Diarize the input",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" and CONFIG.HF_TOKEN != "" else False),
    ),
    min_speakers: Union[int, None] = Query(
        default=None,
        description="Min speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    max_speakers: Union[int, None] = Query(
        default=None,
        description="Max speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]),
):
    result = asr_model.transcribe(
        load_audio(audio_file.file, encode),
        task,
        language,
        initial_prompt,
        vad_filter,
        word_timestamps,
        {"diarize": diarize, "min_speakers": min_speakers, "max_speakers": max_speakers},
        output,
    )
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            "Asr-Engine": CONFIG.ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="{quote(audio_file.filename)}.{output}"',
        },
    )


@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
):
    detected_lang_code, confidence = asr_model.language_detection(load_audio(audio_file.file, encode))
    return {
        "detected_language": tokenizer.LANGUAGES[detected_lang_code],
        "language_code": detected_lang_code,
        "confidence": confidence,
    }


@click.command()
@click.option(
    "-h",
    "--host",
    metavar="HOST",
    default="0.0.0.0",
    help="Host for the webservice (default: 0.0.0.0)",
)
@click.option(
    "-p",
    "--port",
    metavar="PORT",
    default=9000,
    help="Port for the webservice (default: 9000)",
)
@click.version_option(version=projectMetadata["Version"])
def start(host: str, port: Optional[int] = None):
    # Disable uvicorn's default logging - we only use structured logging
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,  # Disable uvicorn's default logging config
        access_log=False,  # Disable access logs
    )


if __name__ == "__main__":
    start()
