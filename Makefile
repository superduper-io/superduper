PYTEST_ARGUMENTS ?= -W ignore
COMPOSE_ARGUMENTS ?=

.PHONY: test-containers
test-containers:
	chmod +x ./tests/material/mongo-init.sh
	docker compose -f tests/material/docker-compose.yml up mongodb mongo-init -d $(COMPOSE_ARGUMENTS)

.PHONY: clean-test-containers
clean-test-containers:
	docker compose -f tests/material/docker-compose.yml down $(COMPOSE_ARGUMENTS)

.PHONY: test
test: test-containers
	black --check superduperdb tests
	ruff check superduperdb tests
	mypy superduperdb
	poetry lock --check
	pytest $(PYTEST_ARGUMENTS)

.PHONY: fix-and-test
fix-and-test: test-containers
	black superduperdb tests
	ruff check --fix superduperdb tests
	mypy superduperdb
	poetry lock --check
	pytest $(PYTEST_ARGUMENTS)

.PHONY: clean-test
clean-test: clean-test-containers
