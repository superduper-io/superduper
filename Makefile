PYTEST_ARGUMENTS ?= -W ignore

dist/.built: pyproject.toml $(shell find superduperdb)
	poetry build --format=wheel
	touch $@

dist: dist/.built

build/requirements.txt: pyproject.toml poetry.lock
	mkdir -p build
	poetry export --without-hashes --format=requirements.txt --output=$@

build/server-image: .dockerignore Dockerfile dist requirements.ci.txt build/requirements.txt
	docker build --target=server -t superduperdb:latest .
	docker image inspect superduperdb:latest -f '{{ .ID }}' > $@

build/jupyter-image: build/server-image
	docker build --target=jupyter -t superduperdb-jupyter:latest .
	docker image inspect superduperdb-jupyter:latest -f '{{ .ID }}' > $@

.PHONY: test-containers
test-containers:
	docker compose -f tests/material/docker-compose.yml up mongodb milvus -d

.PHONY: clean-test-containers
clean-test-containers:
	docker compose -f tests/material/docker-compose.yml down

.PHONY: test
test: test-containers
	black --check superduperdb tests
	ruff check superduperdb tests
	mypy
	poetry lock --check
	$(COVERAGE_PREFIX) pytest $(PYTEST_ARGUMENTS)

.PHONY: fix-and-test
fix-and-test: test-containers
	black superduperdb tests
	ruff check --fix superduperdb tests
	mypy
	poetry lock --check
	$(COVERAGE_PREFIX) pytest $(PYTEST_ARGUMENTS)

.PHONY: clean-test
clean-test: clean-test-containers

.PHONY: jupyter
jupyter: build/jupyter-image
	docker compose -f tests/material/docker-compose.yml up jupyter -d
	# wait until the Jupyter HTTP server responds
	until curl -s -o /dev/null http://127.0.0.1:28888; do sleep 1; done
	open http://127.0.0.1:28888

.PHONY: clean-jupyter
clean-jupyter: clean-test-containers
