PYTEST_ARGUMENTS ?=
COMPOSE_ARGUMENTS ?=
DIRECTORIES = superduperdb test apps

##@ General

# The help target prints out all targets with their descriptions organized
# beneath their categories. The categories are represented by '##@' and the
# target descriptions by '##'.
# https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_parameters
# More info on the awk command:
# http://linuxcommand.org/lc3_adv_awk.php

.DEFAULT_GOAL := help

help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

# RELEASE_VERSION defines the project version for the operator.
# Update this value when you upgrade the version of your project.
# The general flow is VERSION -> make new-release -> GITHUB_ACTIONS -> {make docker_push, ...}
RELEASE_VERSION=$(shell cat VERSION)

new-release: ## Release a new SuperDuperDB version.
	@ if [[ -z "${RELEASE_VERSION}" ]]; then echo "VERSION is not set"; exit 1; fi

	@git add VERSION
	git commit -m "Bump version"
	git tag ${RELEASE_VERSION}


# All these variables are populated after the pushed tag from action "new-release".
TAG_COMMIT := $(shell git rev-list --abbrev-commit --tags --max-count=1)
TAG := $(shell git describe --abbrev=0 --tags ${TAG_COMMIT} 2>/dev/null || true)
COMMIT := $(shell git rev-parse --short HEAD)
DATE := $(shell git log -1 --format=%cd --date=format:"%Y%m%d")
VERSION := $(TAG:v%=%)


##@ Development

fix-and-test: test-containers ## Lint before testing.
	isort $(DIRECTORIES)
	black $(DIRECTORIES)
	ruff check --fix $(DIRECTORIES)
	mypy superduperdb
	pytest $(PYTEST_ARGUMENTS)
	interrogate superduperdb

test-and-fix: test-containers ## Test before linting.
	mypy superduperdb
	pytest $(PYTEST_ARGUMENTS)
	black $(DIRECTORIES)
	ruff check --fix $(DIRECTORIES)
	interrogate superduperdb

lint-and-type-check: ## Apply lint and type checking
	mypy superduperdb
	black --check $(DIRECTORIES)
	ruff check $(DIRECTORIES)
	interrogate superduperdb


##@ Testing

docker-run: docker-build ## Run a SuperDuperDB deployment locally.
	@echo "===> Run SuperDuperDB Locally <==="

	# TODO: make it take as argument the TAG of desired image.
	docker compose -f ./deploy/docker-compose up

test-containers: ## Initialize a local Mongo setup.
	docker compose -f test/material/docker-compose.yml up mongodb mongo-init -d $(COMPOSE_ARGUMENTS)

clean-test-containers: ## Terminate the local Mongo setup.
	docker compose -f test/material/docker-compose.yml down $(COMPOSE_ARGUMENTS)

test: test-containers ## Alias to test-containers
	pytest $(PYTEST_ARGUMENTS)

clean-test: clean-test-containers	## Alias to clean-test-containers


##@ Deployment

docker-build: ## Build SuperDuperDB images.
	@echo "===> Build SuperDuperDB:${TAG} Container <==="
	docker build ./deploy/images/superduperdb  -t superduperdb/superduperdb:${TAG}  --progress=plain

docker-push: docker-build ## Push the latest SuperDuperDB image.
	@echo "===> Set SuperDuperDB:${TAG} as the latest <==="
	docker tag superduperdb/superduperdb:${TAG} superduperdb/superduperdb::latest

	@echo "===> Release SuperDuperDB:${TAG} Container <==="
	docker push superduperdb/superduperdb:${TAG}

	@echo "===> Release SuperDuperDB:latest Container <==="
	docker push superduperdb/superduperdb:latest
