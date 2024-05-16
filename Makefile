DIRECTORIES = superduperdb test
PYTEST_ARGUMENTS ?=
BACKENDS ?= mongodb_community sqlite duckdb pandas

# Default environment file for Pytest
export SUPERDUPERDB_PYTEST_ENV_FILE ?= './deploy/testenv/users.env'

# Export directories for data and artifacts
export SUPERDUPERDB_DATA_DIR ?= ~/.cache/superduperdb/test_data
export SUPERDUPERDB_ARTIFACTS_DIR ?= ~/.cache/superduperdb/artifacts


##@ General

# Display help message with available targets
.DEFAULT_GOAL := help

help: ## Display this help
	@cat ./docs/api/banner.txt
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


##@ Release Management

# Release a new version of SuperDuperDB
# The general flow is VERSION -> make new_release -> GITHUB_ACTIONS -> {make docker_push, ...}
RELEASE_VERSION=$(shell cat VERSION)
CURRENT_RELEASE=$(shell git describe --abbrev=0 --tags)
CURRENT_COMMIT=$(shell git rev-parse --short HEAD)

new_release: ## Release a new version of SuperDuperDB
	@ if [[ -z "${RELEASE_VERSION}" ]]; then echo "VERSION is not set"; exit 1; fi
	@ if [[ "$(RELEASE_VERSION)" == "v$(CURRENT_RELEASE)" ]]; then echo "No new release version. Please update VERSION file."; exit 1; fi
	# Switch to release branch
	@echo "** Switching to branch release-$(RELEASE_VERSION)"
	@git checkout -b release-$(RELEASE_VERSION)
	# Update version in source code
	@echo "** Change superduperdb/__init__.py to version $(RELEASE_VERSION:v%=%)"
	@sed -ie "s/^__version__ = .*/__version__ = '$(RELEASE_VERSION:v%=%)'/" superduperdb/__init__.py
	@git add superduperdb/__init__.py
	# Commit and tag release
	@echo "** Commit Bump Version and Tags"
	@git add VERSION
	@git commit -m "Bump Version $(RELEASE_VERSION:v%=%)"
	@git tag $(RELEASE_VERSION)
	# Push release branch and tag
	@echo "** Push release-$(RELEASE_VERSION)"
	git push --set-upstream origin release-$(RELEASE_VERSION) --tags

install_devkit: ## Add essential development tools
	@echo "Download Docs dependencies"
	python -m pip install --user sphinx furo myst_parser

	@echo "Download Code Quality dependencies"
	python -m pip install --user black==23.3 ruff mypy types-PyYAML types-requests interrogate

	@echo "Download Code Testing dependencies"
	python -m pip install --user pytest pytest-cov "nbval>=0.10.0"


##@ Code Quality

gen_docs: ## Generate Docs and API
	@echo "===> Generate docusaurus docs and blog-posts <==="
	cd docs/hr && npm i --legacy-peer-deps && npm run build
	cd ../..
	@echo "Build finished. The HTML pages are in docs/hr/build"

	@echo "===> Generate Sphinx HTML documentation, including API docs <==="
	rm -rf docs/api/source/
	rm -rf docs/hr/build/apidocs
	sphinx-apidoc -f -o docs/api/source superduperdb
	sphinx-build -a docs/api docs/hr/build/apidocs
	@echo "Build finished. The HTML pages are in docs/hr/build/apidocs"

lint-and-type-check: ## Lint and type-check the code
	@echo "===> Code Formatting <==="
	black --check $(DIRECTORIES)
	ruff check $(DIRECTORIES)

	@echo "===> Static Typing Check <==="

	@if [ -d .mypy_cache ]; then rm -rf .mypy_cache; fi
	mypy superduperdb
	# Check for missing docstrings
	# interrogate superduperdb
	# Check for unused dependencies
	# deptry ./
	# Check for deadcode
	# vulture ./

fix-and-check: ##  Lint the code before testing
	# Code formatting
	black $(DIRECTORIES)
	# Linter and code formatting
	ruff check --fix $(DIRECTORIES)
	# Linting

	@if [ -d .mypy_cache ]; then rm -rf .mypy_cache; fi
	mypy superduperdb


##@ Image Management

# superduperdb/superduperdb is a production image that includes the latest framework from pypi.
# It can be used as "FROM superduper/superduperdb as base" for building custom Dockerfiles.
build_superduperdb: ## Build a minimal Docker image for general use
	echo "===> build superduperdb/superduperdb:$(RELEASE_VERSION:v%=%)"
	docker build . -f ./deploy/images/superduperdb/Dockerfile -t superduperdb/superduperdb:$(RELEASE_VERSION:v%=%) --progress=plain --no-cache \
	--build-arg BUILD_ENV="release"


push_superduperdb: ## Push the superduperdb/superduperdb:<release> image
	@echo "===> release superduperdb/superduperdb:$(RELEASE_VERSION:v%=%)"
	docker push superduperdb/superduperdb:$(RELEASE_VERSION:v%=%)

	@echo "===> release superduperdb/superduperdb:latest"
	docker tag superduperdb/superduperdb:$(RELEASE_VERSION:v%=%) superduperdb/superduperdb:latest
	docker push superduperdb/superduperdb:latest

# superduperdb/sandbox is a development image with all dependencies pre-installed (framework + testenv)
build_sandbox: ## Build superduperdb/sandbox:<commit> image  (RUNNER=<cpu|cuda>)
	# install dependencies.
	python -m pip install toml
	python -c 'import toml; print("\n".join(toml.load(open("pyproject.toml"))["project"]["dependencies"]))' > deploy/testenv/requirements.txt

	# build image
	docker build . -f deploy/images/superduperdb/Dockerfile \
	--build-arg BUILD_ENV="sandbox" \
	--progress=plain \
	--build-arg EXTRA_REQUIREMENTS_FILE="deploy/installations/testenv_requirements.txt" \
	$(if $(RUNNER),--build-arg RUNNER=$(RUNNER),) \
	-t $(if $(filter cuda,$(RUNNER)),superduperdb/sandbox_cuda:$(CURRENT_COMMIT),superduperdb/sandbox:$(CURRENT_COMMIT))

	# mark the image as the latest
	docker tag $(if $(filter cuda,$(RUNNER)),superduperdb/sandbox_cuda:$(CURRENT_COMMIT),superduperdb/sandbox:$(CURRENT_COMMIT)) superduperdb/sandbox:latest


# superduperdb/nightly is a pre-release image with the latest code (and core dependencies) installed.
build_nightly: ## Build superduperdb/nightly:<commit> image (EXTRA_REQUIREMENTS_FILE=<path>) (RUNNER=<cpu|cuda>)
	docker build . -f ./deploy/images/superduperdb/Dockerfile \
	--build-arg BUILD_ENV="nightly" \
	--progress=plain \
	$(if $(EXTRA_REQUIREMENTS_FILE),--build-arg EXTRA_REQUIREMENTS_FILE=$(EXTRA_REQUIREMENTS_FILE),) \
	$(if $(RUNNER),--build-arg RUNNER=$(RUNNER),) \
	-t $(if $(filter cuda,$(RUNNER)),superduperdb/nightly_cuda:$(CURRENT_COMMIT),superduperdb/nightly:$(CURRENT_COMMIT))



push_nightly: ## Push the superduperdb/nightly:<commit> image
	@echo "===> release superduperdb/nightly:$(CURRENT_COMMIT)"
	docker push superduperdb/nightly:$(CURRENT_COMMIT)


##@ Testing Environment

testenv_init: ## Initialize a local Testing environment
	@echo "===> discover superduper/sandbox:latest"
	@if docker image ls superduperdb/sandbox | grep -q "latest"; then \
        echo "superduper/sandbox:latest found";\
    else \
      	echo "superduper/sandbox:latest not found. Please run 'make build_sandbox'";\
      	exit -1;\
    fi

	@echo "===> Discover Hostnames"
	@deploy/testenv/validate_hostnames.sh

	@echo "===> Discover Paths"
	echo "SUPERDUPERDB_DATA_DIR: $(SUPERDUPERDB_DATA_DIR)"
	echo "SUPERDUPERDB_ARTIFACTS_DIR: $(SUPERDUPERDB_ARTIFACTS_DIR)"

	@mkdir -p $(SUPERDUPERDB_DATA_DIR) && chmod -R 777 ${SUPERDUPERDB_DATA_DIR}
	@mkdir -p $(SUPERDUPERDB_ARTIFACTS_DIR) && chmod -R 777 ${SUPERDUPERDB_ARTIFACTS_DIR}
	@mkdir -p deploy/testenv/cache && chmod -R 777 deploy/testenv/cache

	@echo "===> Run TestEnv"
	docker compose -f deploy/testenv/docker-compose.yaml up --remove-orphans &

	@echo "===> Waiting for TestEnv to become ready"
	@cd deploy/testenv/; ./wait_ready.sh

testenv_shutdown: ## Terminate the local Testing environment
	@echo "===> Shutting down the local Testing environment"
	docker compose -f deploy/testenv/docker-compose.yaml down

testenv_restart: testenv_shutdown testenv_init ## Restart the local Testing environment


##@ Database Testing

## Helper function for starting database containers
VALID_DATABASES := mongodb postgres mysql mssql azuresql clickhouse oracle
check_db_variable:
	@if [ -z "$(DB)" ]; then \
		echo "Error: 'DB' is not set."; \
		exit 1; \
	fi; \
	if ! echo "$(VALID_DATABASES)" | grep -qw "$(DB)"; then \
		echo "Error: '$(DB)' is not a valid database name. Valid options are: $(VALID_DATABASES)"; \
		exit 1; \
	fi

testdb_init: check_db_variable ## Init Database Container (DB=<mongodb|postgres|mysql|mssql|azuresql|clickhouse|oracle>)
	@database_path="deploy/databases/$(DB)" && cd "$$database_path" && make init_db

testdb_connect: check_db_variable ## Init Database Container (DB=<mongodb|postgres|mysql|mssql|azuresql|clickhouse|oracle>)
	@database_path="deploy/databases/$(DB)" && cd "$$database_path" && make requirements && make run-example

testdb_shutdown: check_db_variable ## Shutdown Databases Containers (DB=<mongodb|postgres|mysql|mssql|azuresql|clickhouse|oracle>)
	@database_path="deploy/databases/$(DB)" && cd "$$database_path" && make shutdown_db

##@ CI Testing Functions

unit_testing: ## Execute unit testing
	pytest $(PYTEST_ARGUMENTS) ./test/unittest/

databackend_testing: ## Execute integration testing
	@echo "TESTING BACKENDS"
	@for backend in $(BACKENDS); do \
		echo "TESTING $$backend"; \
		SUPERDUPERDB_CONFIG=deploy/testenv/env/integration/backends/$$backend.yaml pytest $(PYTEST_ARGUMENTS) ./test/integration/backends; \
	done
	@echo "TODO -- implement more backends integration testing..."

ext_testing: ## Execute integration testing
	find ./test -type d -name __pycache__ -exec rm -r {} +
	find ./test -type f -name "*.pyc" -delete
	pytest $(PYTEST_ARGUMENTS) ./test/integration/ext

smoke_testing: ## Execute smoke testing
	SUPERDUPERDB_CONFIG=deploy/testenv/env/smoke/config.yaml pytest $(PYTEST_ARGUMENTS) ./test/smoke