PYTEST_ARGUMENTS ?=
COMPOSE_ARGUMENTS ?=

.PHONY: test-containers
test-containers:
	docker compose -f tests/material/docker-compose.yml up mongodb mongo-init -d $(COMPOSE_ARGUMENTS)

.PHONY: clean-test-containers
clean-test-containers:
	docker compose -f tests/material/docker-compose.yml down $(COMPOSE_ARGUMENTS)

.PHONY: lint-and-type-check
lint-and-type-check:
	black --check superduperdb tests
	ruff check superduperdb tests
	mypy superduperdb

.PHONY: test
test: test-containers
	pytest $(PYTEST_ARGUMENTS)

.PHONY: fix-and-test
fix-and-test: test-containers
	black superduperdb tests
	ruff check --fix superduperdb tests
	mypy superduperdb
	pytest $(PYTEST_ARGUMENTS)

.PHONY: clean-test
clean-test: clean-test-containers
