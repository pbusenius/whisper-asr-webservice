![Release](https://img.shields.io/github/v/release/pbusenius/whisper-asr-webservice.svg)
![Docker Pulls](https://img.shields.io/docker/pulls/pbusenius/openai-whisper-asr-webservice.svg)
![Build](https://img.shields.io/github/actions/workflow/status/pbusenius/whisper-asr-webservice/docker-publish.yml.svg)
![Licence](https://img.shields.io/github/license/pbusenius/whisper-asr-webservice.svg)

# ASR-API

ASR-API is a general-purpose speech recognition toolkit. Whisper Models are trained on a large dataset of diverse audio and is also a multitask model that can perform multilingual speech recognition as well as speech translation and language identification.

## Features

Current release (v1.10.0-dev) supports following whisper models:

- [openai/whisper](https://github.com/openai/whisper)@[v20250625](https://github.com/openai/whisper/releases/tag/v20250625)
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)@[v1.1.1](https://github.com/SYSTRAN/faster-whisper/releases/tag/v1.1.1)
- [whisperX](https://github.com/m-bain/whisperX)@[v3.4.2](https://github.com/m-bain/whisperX/releases/tag/v3.4.2)
- [Voxtral](https://voxtral.life/) (Mistral AI)

## Quick Usage

### CPU

```shell
docker run -d -p 9000:9000 \
  -e ASR_MODEL=base \
  -e ASR_ENGINE=openai_whisper \
  pbusenius/openai-whisper-asr-webservice:latest
```

### GPU

```shell
docker run -d --gpus all -p 9000:9000 \
  -e ASR_MODEL=base \
  -e ASR_ENGINE=openai_whisper \
  pbusenius/openai-whisper-asr-webservice:latest-gpu
```

#### Cache

To reduce container startup time by avoiding repeated downloads, you can persist the cache directory:

```shell
docker run -d -p 9000:9000 \
  -v $PWD/cache:/root/.cache/ \
  pbusenius/openai-whisper-asr-webservice:latest
```

## Key Features

- Multiple ASR engines support (OpenAI Whisper, Faster Whisper, WhisperX, Voxtral)
- Multiple output formats (text, JSON, VTT, SRT, TSV)
- Word-level timestamps support
- Voice activity detection (VAD) filtering
- Speaker diarization (with WhisperX)
- FFmpeg integration for broad audio/video format support
- GPU acceleration support
- Configurable model loading/unloading
- REST API with Swagger documentation
- OpenTelemetry monitoring and observability
- Health check endpoint (`/health`)
- Prometheus metrics endpoint (`/metrics`)

## Environment Variables

Key configuration options:

- `ASR_ENGINE`: Engine selection (openai_whisper, faster_whisper, whisperx, voxtral)
- `ASR_MODEL`: Model selection (tiny, base, small, medium, large-v3, etc. or Voxtral-Mini-3B-2507 for Voxtral)
- `ASR_MODEL_PATH`: Custom path to store/load models
- `ASR_DEVICE`: Device selection (cuda, cpu)
- `MODEL_IDLE_TIMEOUT`: Timeout for model unloading
- `OTEL_ENABLED`: Enable/disable OpenTelemetry instrumentation (default: true)
- `OTEL_SERVICE_NAME`: Service name for OpenTelemetry (default: asr-api)
- `OTEL_EXPORTER_OTLP_ENDPOINT`: Optional OTLP endpoint for traces/metrics
- `OTEL_PROMETHEUS_METRICS_PATH`: Path for Prometheus metrics endpoint (default: /metrics)
- `OTEL_LOG_LEVEL`: Set to "debug" to enable console span exporter for debugging

## OpenTelemetry Konfiguration

OpenTelemetry ist standardmäßig aktiviert und bietet automatisches Monitoring der API. Es gibt verschiedene Konfigurationsmöglichkeiten:

### Standard-Konfiguration (Prometheus Metrics)

Standardmäßig werden Prometheus-Metriken am `/metrics` Endpoint bereitgestellt:

```shell
docker run -d -p 9000:9000 \
  -e ASR_MODEL=base \
  pbusenius/asr-api:latest
```

Die Metriken können dann von Prometheus gescrapt werden:
```yaml
scrape_configs:
  - job_name: 'asr-api'
    static_configs:
      - targets: ['localhost:9000']
```

### OTLP Exporter (für Jaeger, Tempo, etc.)

Um Traces und Metriken an einen OTLP Collector zu senden:

```shell
docker run -d -p 9000:9000 \
  -e ASR_MODEL=base \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317 \
  pbusenius/asr-api:latest
```

**Hinweis:** Für OTLP Support muss das optionale Paket `opentelemetry-exporter-otlp` installiert sein:

```shell
# Mit uv
uv sync --extra otlp

# Oder mit pip
pip install opentelemetry-exporter-otlp
```

Falls es fehlt, wird eine Warnung ausgegeben und OTLP wird nicht verwendet.

### Debug-Modus (Console Exporter)

Für lokale Entwicklung können Traces in der Konsole ausgegeben werden:

```shell
docker run -d -p 9000:9000 \
  -e ASR_MODEL=base \
  -e OTEL_LOG_LEVEL=debug \
  pbusenius/asr-api:latest
```

### OpenTelemetry deaktivieren

Falls OpenTelemetry nicht benötigt wird:

```shell
docker run -d -p 9000:9000 \
  -e ASR_MODEL=base \
  -e OTEL_ENABLED=false \
  pbusenius/asr-api:latest
```

### Benutzerdefinierter Service-Name

```shell
docker run -d -p 9000:9000 \
  -e ASR_MODEL=base \
  -e OTEL_SERVICE_NAME=my-asr-service \
  pbusenius/asr-api:latest
```

### Vollständige Beispiel-Konfiguration

```shell
docker run -d -p 9000:9000 \
  -e ASR_MODEL=base \
  -e ASR_ENGINE=faster_whisper \
  -e OTEL_ENABLED=true \
  -e OTEL_SERVICE_NAME=asr-api-production \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317 \
  -e OTEL_PROMETHEUS_METRICS_PATH=/metrics \
  pbusenius/asr-api:latest
```

## Development

```shell
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies for cpu
uv sync --extra cpu

# Install dependencies for cuda (GPU)
uv sync --extra cuda --extra-index-url https://download.pytorch.org/whl/cu126

# Run service (CPU)
uv run asr-api --host 0.0.0.0 --port 9000

# Run service (GPU/CUDA)
uv run --extra cuda --extra-index-url https://download.pytorch.org/whl/cu126 asr-api --host 0.0.0.0 --port 9000
```

After starting the service, visit `http://localhost:9000` or `http://0.0.0.0:9000` in your browser to access the Swagger UI documentation and try out the API endpoints.

## TODO

Observability improvements:

- [ ] Structured logging (replace print statements with proper logging infrastructure)
- [x] Health check endpoints (`/health`, `/ready`)
- [x] Prometheus metrics endpoint (`/metrics`)
- [x] Request duration tracking

New ASR models:

- [x] Add Voxtral ASR model support

## Credits

- This software uses libraries from the [FFmpeg](http://ffmpeg.org) project under the [LGPLv2.1](http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)
