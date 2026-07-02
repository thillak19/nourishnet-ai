# Makefile for common tasks
.PHONY: help build bootstrap train run

help:
	@echo "Makefile commands:"
	@echo "  make build     - build Docker image"
	@echo "  make bootstrap - generate data and create minimal model artifacts"
	@echo "  make run       - run the API locally (uvicorn)"
	@echo "  make docker-run - run using docker-compose"

build:
	docker build -t nourishnet-ai:latest .

bootstrap:
	python scripts/bootstrap_models.py

run:
	uvicorn src.api.main:app --reload

docker-run:
	docker-compose up --build
