FROM swaggerapi/swagger-ui:v5.9.1 AS swagger-ui

FROM python:3.10-bookworm

LABEL org.opencontainers.image.source="https://github.com/pbusenius/asr-api"

WORKDIR /app

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY . .
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui.css swagger-ui-assets/swagger-ui.css
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui-bundle.js swagger-ui-assets/swagger-ui-bundle.js

RUN uv sync --extra cpu

EXPOSE 9000

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["asr-api"]
