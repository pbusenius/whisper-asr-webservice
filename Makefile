.PHONY: build push clean clean-images clean-containers help

# Variables
VERSION := 0.2.0
IMAGE_NAME := pbusenius/asr-api
IMAGE_TAG := $(IMAGE_NAME):$(VERSION)
IMAGE_LATEST := $(IMAGE_NAME):latest

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build GPU Docker image
	@echo "Building GPU image: $(IMAGE_TAG)"
	docker build -f Dockerfile --build-arg ASR_MODEL=base --build-arg ASR_ENGINE=faster_whisper -t $(IMAGE_TAG) -t $(IMAGE_LATEST) .

push: ## Push Docker images
	@echo "Pushing images: $(IMAGE_TAG) and $(IMAGE_LATEST)"
	docker push $(IMAGE_TAG)
	docker push $(IMAGE_LATEST)

clean: clean-images clean-containers ## Clean Docker images and containers

clean-images: ## Remove Docker images
	@echo "Removing Docker images..."
	-docker rmi $(IMAGE_TAG) $(IMAGE_LATEST) 2>/dev/null || true

clean-containers: ## Remove stopped containers
	@echo "Removing stopped containers..."
	-docker container prune -f

clean-all: clean-images clean-containers ## Remove images, containers, and volumes
	@echo "Removing all Docker resources..."
	-docker system prune -a -f --volumes

