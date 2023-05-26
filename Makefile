
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

.PHONY: test
test:
	docker compose -f tests/material/docker-compose.yml up mongodb -d
	pytest -vv --maxfail=3 tests/unittests

.PHONY: clean-test
clean-test:
	docker compose -f tests/material/docker-compose.yml down

.PHONY: jupyter
jupyter: build/jupyter-image
	docker compose -f tests/material/docker-compose.yml up jupyter -d
	# wait until the Jupyter HTTP server responds
	until curl -s -o /dev/null http://127.0.0.1:28888; do sleep 1; done
	open http://127.0.0.1:28888

.PHONY: clean-jupyter
clean-jupyter:
	docker compose -f tests/material/docker-compose.yml down
