PYTEST_ARGUMENTS ?=
DIRECTORIES = superduperdb test apps

##@ General

# The help target prints out all targets with their descriptions organized
# beneath their categories. The categories are represented by '##@' and the
# target descriptions by '##'.
# https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_parameters
# More info on the awk command:
# http://linuxcommand.org/lc3_adv_awk.php

.DEFAULT_GOAL := help

help: ## Display this help
	@cat ./apidocs/banner.txt

	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


# RELEASE_VERSION defines the project version for the operator.
# Update this value when you upgrade the version of your project.
# The general flow is VERSION -> make new-release -> GITHUB_ACTIONS -> {make docker_push, ...}
RELEASE_VERSION=$(shell cat VERSION)

# All these variables are populated after the pushed tag from action "new-release".
TAG_COMMIT := $(shell git rev-list --abbrev-commit --tags --max-count=1)
TAG := $(shell git describe --abbrev=0 --tags ${TAG_COMMIT} 2>/dev/null || true)
COMMIT := $(shell git rev-parse --short HEAD)
DATE := $(shell git log -1 --format=%cd --date=format:"%Y%m%d")
VERSION := $(TAG:v%=%)


##@ Release Management

docker-build: ## Build SuperDuperDB images
	@echo "===> Build SuperDuperDB:${TAG} Container <==="
	docker build ./deploy/images/superduperdb  -t superduperdb/superduperdb:${TAG}  --progress=plain


docker-push: ## Push the latest SuperDuperDB image
	@echo "===> Set SuperDuperDB:${TAG} as the latest <==="
	docker tag superduperdb/superduperdb:${TAG} superduperdb/superduperdb:latest

	@echo "===> Release SuperDuperDB:${TAG} Container <==="
	docker push superduperdb/superduperdb:${TAG}

	@echo "===> Release SuperDuperDB:latest Container <==="
	docker push superduperdb/superduperdb:latest

new-release: ## Release a new SuperDuperDB version
	@ if [[ -z "${RELEASE_VERSION}" ]]; then echo "VERSION is not set"; exit 1; fi
	@ if [[ "${RELEASE_VERSION}" == "${TAG}" ]]; then echo "no new release version. Please update VERSION file."; exit 1; fi

	@echo "** Change superduperdb/__init__.py to version $(RELEASE_VERSION:v%=%)"
	@sed -ie "s/^__version__ = .*/__version__ = '$(RELEASE_VERSION:v%=%)'/" superduperdb/__init__.py

	@echo "** Commit Changes"
	@git add VERSION
	git commit -m "Bump version"

	@echo "** Push tag for version $(RELEASE_VERSION:v%=%)"
	@git tag ${RELEASE_VERSION}


##@ Development

lint-and-type-check: ## Apply lint and type checking
	mypy superduperdb
	black --check $(DIRECTORIES)
	ruff check $(DIRECTORIES)
	interrogate superduperdb

fix-and-test: local_mongo_init ## Lint before testing
	isort $(DIRECTORIES)
	black $(DIRECTORIES)
	ruff check --fix $(DIRECTORIES)
	mypy superduperdb
	pytest $(PYTEST_ARGUMENTS)
	interrogate superduperdb

local_mongo_init: ## Initialize a local MongoDB setup
	docker compose -f test/material/docker-compose.yml up mongodb mongo-init -d $(COMPOSE_ARGUMENTS)

local_mongo_shutdown: ## Terminate the local MongoDB setup
	docker compose -f test/material/docker-compose.yml down $(COMPOSE_ARGUMENTS)

debug_env:  ## Run Jupyter with the local version of SuperDuper
	docker run -p 8888:8888 -v ./superduperdb:/home/superduper/superduperdb superduperdb/superduperdb:latest

##@ Demo

demo-run: ## Run a SuperDuperDB demo locally
	@echo "===> Run SuperDuperDB Locally <==="

	# TODO: make it take as argument the TAG of desired image.
	docker compose -f ./deploy/docker-compose/demo.yaml up

