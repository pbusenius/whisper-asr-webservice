.PHONY: build build-cpu build-gpu push push-cpu push-gpu push-all clean clean-images clean-containers help

# Variables
VERSION := 0.1
IMAGE_NAME := pbusenius/asr-api
CPU_TAG := $(IMAGE_NAME):$(VERSION)
CPU_LATEST := $(IMAGE_NAME):latest
GPU_TAG := $(IMAGE_NAME):$(VERSION)-gpu
GPU_LATEST := $(IMAGE_NAME):latest-gpu

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: build-cpu build-gpu ## Build both CPU and GPU Docker images

build-cpu: ## Build CPU Docker image
	@echo "Building CPU image: $(CPU_TAG)"
	docker build -f Dockerfile -t $(CPU_TAG) -t $(CPU_LATEST) .

build-gpu: ## Build GPU Docker image
	@echo "Building GPU image: $(GPU_TAG)"
	docker build -f Dockerfile.gpu -t $(GPU_TAG) -t $(GPU_LATEST) .

push: push-all ## Push both CPU and GPU Docker images

push-cpu: ## Push CPU Docker images
	@echo "Pushing CPU images: $(CPU_TAG) and $(CPU_LATEST)"
	docker push $(CPU_TAG)
	docker push $(CPU_LATEST)

push-gpu: ## Push GPU Docker images
	@echo "Pushing GPU images: $(GPU_TAG) and $(GPU_LATEST)"
	docker push $(GPU_TAG)
	docker push $(GPU_LATEST)

push-all: push-cpu push-gpu ## Push all Docker images

clean: clean-images clean-containers ## Clean Docker images and containers

clean-images: ## Remove Docker images
	@echo "Removing Docker images..."
	-docker rmi $(CPU_TAG) $(CPU_LATEST) $(GPU_TAG) $(GPU_LATEST) 2>/dev/null || true

clean-containers: ## Remove stopped containers
	@echo "Removing stopped containers..."
	-docker container prune -f

clean-all: clean-images clean-containers ## Remove images, containers, and volumes
	@echo "Removing all Docker resources..."
	-docker system prune -a -f --volumes

