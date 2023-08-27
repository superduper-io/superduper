PYTEST_ARGUMENTS ?=
COMPOSE_ARGUMENTS ?=

.PHONY: test-containers
test-containers:
	docker compose -f test/material/docker-compose.yml up mongodb mongo-init -d $(COMPOSE_ARGUMENTS)

.PHONY: clean-test-containers
clean-test-containers:
	docker compose -f test/material/docker-compose.yml down $(COMPOSE_ARGUMENTS)

.PHONY: lint-and-type-check
lint-and-type-check:
	black --check superduperdb test
	ruff check superduperdb test
	mypy superduperdb
	interrogate superduperdb

.PHONY: test
test: test-containers
	pytest $(PYTEST_ARGUMENTS)

.PHONY: fix-and-test
fix-and-test: test-containers fix-codestyle
	black superduperdb test
	ruff check --fix superduperdb test
	mypy superduperdb
	pytest $(PYTEST_ARGUMENTS)
	interrogate superduperdb

.PHONY: fix-codestyle
fix-codestyle:
	black superduperdb test
	ruff check --fix superduperdb test
	mypy superduperdb

.PHONY: clean-test
clean-test: clean-test-containers
