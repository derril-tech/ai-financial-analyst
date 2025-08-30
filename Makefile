# AI Financial Analyst - Development Commands

.PHONY: help up down build clean seed test lint format install

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "Installing dependencies..."
	pnpm install
	cd apps/api && poetry install

up: ## Start all services with docker-compose
	@echo "Starting services..."
	docker-compose up -d

down: ## Stop all services
	@echo "Stopping services..."
	docker-compose down

build: ## Build all services
	@echo "Building services..."
	docker-compose build

clean: ## Clean up containers and volumes
	@echo "Cleaning up..."
	docker-compose down -v --remove-orphans
	docker system prune -f

seed: ## Seed the database with initial data
	@echo "Seeding database..."
	cd apps/api && poetry run alembic upgrade head

test: ## Run tests
	@echo "Running tests..."
	cd apps/api && poetry run pytest
	cd apps/web && pnpm test

lint: ## Run linters
	@echo "Running linters..."
	cd apps/api && poetry run black --check . && poetry run isort --check-only . && poetry run ruff check .
	cd apps/web && pnpm lint
	pnpm format:check

format: ## Format code
	@echo "Formatting code..."
	cd apps/api && poetry run black . && poetry run isort .
	pnpm format

dev-api: ## Start API in development mode
	@echo "Starting API in development mode..."
	cd apps/api && poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-web: ## Start web app in development mode
	@echo "Starting web app in development mode..."
	cd apps/web && pnpm dev

logs: ## Show logs from all services
	docker-compose logs -f

logs-api: ## Show API logs
	docker-compose logs -f api

logs-web: ## Show web logs
	docker-compose logs -f web

shell-api: ## Open shell in API container
	docker-compose exec api bash

shell-db: ## Open database shell
	docker-compose exec postgres psql -U postgres -d ai_financial_analyst

backup-db: ## Backup database
	@echo "Backing up database..."
	docker-compose exec postgres pg_dump -U postgres ai_financial_analyst > backup_$(shell date +%Y%m%d_%H%M%S).sql

restore-db: ## Restore database from backup (usage: make restore-db FILE=backup.sql)
	@echo "Restoring database from $(FILE)..."
	docker-compose exec -T postgres psql -U postgres ai_financial_analyst < $(FILE)
