PYTEST_ARGUMENTS ?=
COMPOSE_ARGUMENTS ?=
DIRECTORIES = superduperdb test apps

.PHONY: test-containers
test-containers:
	docker compose -f test/material/docker-compose.yml up mongodb mongo-init -d $(COMPOSE_ARGUMENTS)

.PHONY: clean-test-containers
clean-test-containers:
	docker compose -f test/material/docker-compose.yml down $(COMPOSE_ARGUMENTS)

.PHONY: lint-and-type-check
lint-and-type-check:
	isort --check $(DIRECTORIES)
	black --check $(DIRECTORIES)
	ruff check $(DIRECTORIES)
	mypy superduperdb
	interrogate superduperdb

.PHONY: test
test: test-containers
	pytest $(PYTEST_ARGUMENTS)

.PHONY: fix-and-test
fix-and-test: test-containers
	isort $(DIRECTORIES)
	black $(DIRECTORIES)
	ruff check --fix $(DIRECTORIES)
	mypy superduperdb
	pytest $(PYTEST_ARGUMENTS)
	interrogate superduperdb

.PHONY: clean-test
clean-test: clean-test-containers
