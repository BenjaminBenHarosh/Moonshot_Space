.PHONY: help install test lint format clean build run docker-build docker-run docker-stop

# Default target
help:
	@echo "Capsule Acceleration Simulator"
	@echo "Available commands:"
	@echo "  install       Install dependencies"
	@echo "  test          Run tests"
	@echo "  lint          Run linting"
	@echo "  format        Format code"
	@echo "  clean         Clean up generated files"
	@echo "  run           Run API server locally"
	@echo "  demo          Run example scenarios"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run with Docker Compose"
	@echo "  docker-stop   Stop Docker services"

# Development setup
install:
	pip install -r requirements.txt

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html

# Code quality
lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# Cleanup
clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf outputs/*.parquet
	rm -rf outputs/*.json*
	rm -rf outputs/*.h5
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Local development
run:
	uvicorn api:app --reload --host 0.0.0.0 --port 8000

demo:
	python example_run.py

# Docker operations
docker-build:
	docker build -t capsule-simulator .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f simulator

# Production deployment
deploy: docker-build
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Create sample configs
sample-configs:
	mkdir -p configs
	python -c "from config import DEFAULT_ASSIGNMENT_CONFIG; DEFAULT_ASSIGNMENT_CONFIG.to_yaml('configs/default.yaml')"
	@echo "Sample configuration created in configs/default.yaml"

# Database operations (if using PostgreSQL)
db-init:
	docker-compose exec postgres psql -U postgres -d simulator -f /docker-entrypoint-initdb.d/init.sql

db-reset:
	docker-compose down postgres
	docker volume rm $(shell basename $(CURDIR))_postgres_data
	docker-compose up -d postgres
