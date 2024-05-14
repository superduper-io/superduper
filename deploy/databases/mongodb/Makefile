.DEFAULT_GOAL := help

help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


init_db: ## Init database on Docker
	docker compose up --remove-orphans  &

shutdown_db: ## Init database on Docker
	docker compose down


requirements: ## Install client requirements
	pip install -r requirements.txt

run-example: ## Run the example
	./wait_ready.sh
	python example.py