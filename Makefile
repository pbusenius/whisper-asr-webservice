.PHONY: build build-faster-whisper build-openai-whisper build-whisperx build-parakeet build-voxtral build-vllm-whisper push push-all clean clean-images clean-containers help

# Variables
VERSION := 0.2.0
IMAGE_NAME := pbusenius/asr-api

# Engine-specific image names
FASTER_WHISPER_TAG := $(IMAGE_NAME)-faster-whisper:$(VERSION)
FASTER_WHISPER_LATEST := $(IMAGE_NAME)-faster-whisper:latest
OPENAI_WHISPER_TAG := $(IMAGE_NAME)-openai-whisper:$(VERSION)
OPENAI_WHISPER_LATEST := $(IMAGE_NAME)-openai-whisper:latest
WHISPERX_TAG := $(IMAGE_NAME)-whisperx:$(VERSION)
WHISPERX_LATEST := $(IMAGE_NAME)-whisperx:latest
PARAKEET_TAG := $(IMAGE_NAME)-parakeet:$(VERSION)
PARAKEET_LATEST := $(IMAGE_NAME)-parakeet:latest
VOXTRAL_TAG := $(IMAGE_NAME)-voxtral:$(VERSION)
VOXTRAL_LATEST := $(IMAGE_NAME)-voxtral:latest
VLLM_WHISPER_TAG := $(IMAGE_NAME)-vllm-whisper:$(VERSION)
VLLM_WHISPER_LATEST := $(IMAGE_NAME)-vllm-whisper:latest

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-25s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: build-faster-whisper ## Build all Docker images (default: faster-whisper)

build-faster-whisper: ## Build Faster Whisper Docker image
	@echo "Building Faster Whisper image: $(FASTER_WHISPER_TAG)"
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.faster-whisper --build-arg ASR_MODEL=base -t $(FASTER_WHISPER_TAG) -t $(FASTER_WHISPER_LATEST) .

build-openai-whisper: ## Build OpenAI Whisper Docker image
	@echo "Building OpenAI Whisper image: $(OPENAI_WHISPER_TAG)"
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.openai-whisper --build-arg ASR_MODEL=base -t $(OPENAI_WHISPER_TAG) -t $(OPENAI_WHISPER_LATEST) .

build-whisperx: ## Build WhisperX Docker image
	@echo "Building WhisperX image: $(WHISPERX_TAG)"
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.whisperx --build-arg ASR_MODEL=base -t $(WHISPERX_TAG) -t $(WHISPERX_LATEST) .

build-parakeet: ## Build Parakeet Docker image
	@echo "Building Parakeet image: $(PARAKEET_TAG)"
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.parakeet --build-arg ASR_MODEL=parakeet-tdt-0.6b-v3 -t $(PARAKEET_TAG) -t $(PARAKEET_LATEST) .

build-voxtral: ## Build Voxtral Docker image
	@echo "Building Voxtral image: $(VOXTRAL_TAG)"
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.voxtral --build-arg ASR_MODEL=Voxtral-Mini-3B-2507 -t $(VOXTRAL_TAG) -t $(VOXTRAL_LATEST) .

build-vllm-whisper: ## Build vLLM Whisper Docker image
	@echo "Building vLLM Whisper image: $(VLLM_WHISPER_TAG)"
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.vllm-whisper --build-arg ASR_MODEL=openai/whisper-large-v3 -t $(VLLM_WHISPER_TAG) -t $(VLLM_WHISPER_LATEST) .

push-all: push-faster-whisper push-openai-whisper push-whisperx push-parakeet push-voxtral push-vllm-whisper ## Push all Docker images

push-faster-whisper: ## Push Faster Whisper images
	docker push $(FASTER_WHISPER_TAG)
	docker push $(FASTER_WHISPER_LATEST)

push-openai-whisper: ## Push OpenAI Whisper images
	docker push $(OPENAI_WHISPER_TAG)
	docker push $(OPENAI_WHISPER_LATEST)

push-whisperx: ## Push WhisperX images
	docker push $(WHISPERX_TAG)
	docker push $(WHISPERX_LATEST)

push-parakeet: ## Push Parakeet images
	docker push $(PARAKEET_TAG)
	docker push $(PARAKEET_LATEST)

push-voxtral: ## Push Voxtral images
	docker push $(VOXTRAL_TAG)
	docker push $(VOXTRAL_LATEST)

push-vllm-whisper: ## Push vLLM Whisper images
	docker push $(VLLM_WHISPER_TAG)
	docker push $(VLLM_WHISPER_LATEST)

push: push-faster-whisper ## Push Docker images (default: faster-whisper)

clean: clean-images clean-containers ## Clean Docker images and containers

clean-images: ## Remove Docker images
	@echo "Removing Docker images..."
	-docker rmi $(FASTER_WHISPER_TAG) $(FASTER_WHISPER_LATEST) 2>/dev/null || true
	-docker rmi $(OPENAI_WHISPER_TAG) $(OPENAI_WHISPER_LATEST) 2>/dev/null || true
	-docker rmi $(WHISPERX_TAG) $(WHISPERX_LATEST) 2>/dev/null || true
	-docker rmi $(PARAKEET_TAG) $(PARAKEET_LATEST) 2>/dev/null || true
	-docker rmi $(VOXTRAL_TAG) $(VOXTRAL_LATEST) 2>/dev/null || true
	-docker rmi $(VLLM_WHISPER_TAG) $(VLLM_WHISPER_LATEST) 2>/dev/null || true

clean-containers: ## Remove stopped containers
	@echo "Removing stopped containers..."
	-docker container prune -f

clean-all: clean-images clean-containers ## Remove images, containers, and volumes
	@echo "Removing all Docker resources..."
	-docker system prune -a -f --volumes

