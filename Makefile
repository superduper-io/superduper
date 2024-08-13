DIRECTORIES ?= superduper test
SUPERDUPER_CONFIG ?= test/configs/default.yaml
PYTEST_ARGUMENTS ?=
PLUGIN_NAME ?=

# Export directories for data and artifacts
export SUPERDUPER_DATA_DIR ?= ~/.cache/superduper/test_data
export SUPERDUPER_ARTIFACTS_DIR ?= ~/.cache/superduper/artifacts


##@ General

# Display help message with available targets
.DEFAULT_GOAL := help

help: ## Display this help
	@cat ./docs/static/banner.txt
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


##@ Release Management

# Release a new version of superduper.io
# The general flow is VERSION -> make new_release -> GITHUB_ACTIONS -> {make docker_push, ...}
RELEASE_VERSION=$(shell cat VERSION)
CURRENT_RELEASE=$(shell git describe --abbrev=0 --tags)
CURRENT_COMMIT=$(shell git rev-parse --short HEAD)

new_release: ## Release a new version of superduper.io
	@ if [[ -z "${RELEASE_VERSION}" ]]; then echo "VERSION is not set"; exit 1; fi
	@ if [[ "$(RELEASE_VERSION)" == "v$(CURRENT_RELEASE)" ]]; then echo "No new release version. Please update VERSION file."; exit 1; fi
	# Switch to release branch
	@echo "** Switching to branch release-$(RELEASE_VERSION)"
	@git checkout -b release-$(RELEASE_VERSION)
	# Update version in source code
	@echo "** Change superduper/__init__.py to version $(RELEASE_VERSION)"
	@sed -ie "s/^__version__ = .*/__version__ = '$(RELEASE_VERSION:v%=%)'/" superduper/__init__.py
	@git add superduper/__init__.py
	# Commit and tag release
	@echo "** Commit Bump Version and Tags"
	@git add VERSION CHANGELOG.md
	@git commit -m "Bump Version $(RELEASE_VERSION)"
	@git tag $(RELEASE_VERSION)

	# Push branch and set upstream
	git push --set-upstream origin release-$(RELEASE_VERSION)

	# Push the specific tag
	git push origin $(RELEASE_VERSION)

##@ Code Quality

gen_docs: ## Generate Docs and API
	@echo "===> Generate docusaurus docs and blog-posts <==="
	cd docs && npm i --legacy-peer-deps && npm run build
	cd ..
	@echo "Build finished. The HTML pages are in docs/hr/build"

lint-and-type-check: ## Lint and type-check the code
	@echo "===> Code Formatting <==="
	black --check $(DIRECTORIES)
	ruff check $(DIRECTORIES)

	@echo "===> Static Typing Check <==="

	@if [ -d .mypy_cache ]; then rm -rf .mypy_cache; fi
	mypy superduper
	# Check for missing docstrings
	# interrogate superduper
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
	mypy superduper


##@ Image Management

# superduperio/superduper is a production image that includes the latest framework from pypi.
# It can be used as "FROM superduperio/superduper as base" for building custom Dockerfiles.
build_superduper: ## Build a minimal Docker image for general use
	echo "===> build superduperio/superduper:$(RELEASE_VERSION:v%=%)"
	docker build . -f ./deploy/images/superduper/Dockerfile -t superduperio/superduper:$(RELEASE_VERSION:v%=%) --progress=plain --no-cache \
	--build-arg BUILD_ENV="superduper"


push_superduper: ## Push the superduperio/superduper:<release> image
	@echo "===> release superduperio/superduper:$(RELEASE_VERSION:v%=%)"
	docker push superduperio/superduper:$(RELEASE_VERSION:v%=%)

	@echo "===> release superduperio/superduper:latest"
	docker tag superduperio/superduper:$(RELEASE_VERSION:v%=%) superduperio/superduper:latest
	docker push superduperio/superduper:latest

# superduperio/sandbox is a development image with all dependencies pre-installed (framework + testenv)
build_sandbox: ## Build superduperio/sandbox:<commit> image  (RUNNER=<cpu|cuda>)
	# install dependencies.
	python -m pip install toml
	python -c 'import toml; print("\n".join(toml.load(open("pyproject.toml"))["project"]["dependencies"]))' > deploy/testenv/requirements.txt

	# build image
	docker build . -f deploy/images/superduper/Dockerfile \
	--build-arg BUILD_ENV="sandbox" \
	--progress=plain \
	--build-arg EXTRA_REQUIREMENTS_FILE="deploy/installations/testenv_requirements.txt" \
	$(if $(RUNNER),--build-arg RUNNER=$(RUNNER),) \
	-t $(if $(filter cuda,$(RUNNER)),superduperio/sandbox_cuda:$(CURRENT_COMMIT),superduperio/sandbox:$(CURRENT_COMMIT))

	# mark the image as the latest
	docker tag $(if $(filter cuda,$(RUNNER)),superduperio/sandbox_cuda:$(CURRENT_COMMIT),superduperio/sandbox:$(CURRENT_COMMIT)) superduperio/sandbox:latest


# superduperio/nightly is a pre-release image with the latest code (and core dependencies) installed.
build_nightly: ## Build superduperio/nightly:<commit> image (EXTRA_REQUIREMENTS_FILE=<path>) (RUNNER=<cpu|cuda>)
	docker build . -f ./deploy/images/superduper/Dockerfile \
	--build-arg BUILD_ENV="nightly" \
    --platform linux/amd64 \
	--progress=plain \
	$(if $(EXTRA_REQUIREMENTS_FILE),--build-arg EXTRA_REQUIREMENTS_FILE=$(EXTRA_REQUIREMENTS_FILE),) \
	$(if $(RUNNER),--build-arg RUNNER=$(RUNNER),) \
	-t $(if $(filter cuda,$(RUNNER)),superduperio/nightly_cuda:$(CURRENT_COMMIT),superduperio/nightly:$(CURRENT_COMMIT))


push_nightly: ## Push the superduperio/nightly:<commit> image
	@echo "===> release superduperio/nightly:$(CURRENT_COMMIT)"
	docker push superduperio/nightly:$(CURRENT_COMMIT)


##@ Testing Environment

testenv_init: ## Initialize a local Testing environment
	@echo "===> discover superduperio/sandbox:latest"
	@if docker image ls superduperio/sandbox | grep -q "latest"; then \
        echo "superduperio/sandbox:latest found";\
    else \
      	echo "superduperio/sandbox:latest not found. Please run 'make build_sandbox'";\
      	exit -1;\
    fi

	@echo "===> Discover Hostnames"
	@deploy/testenv/validate_hostnames.sh

	@echo "===> Discover Paths"
	echo "SUPERDUPER_DATA_DIR: $(SUPERDUPER_DATA_DIR)"
	echo "SUPERDUPER_ARTIFACTS_DIR: $(SUPERDUPER_ARTIFACTS_DIR)"

	@mkdir -p $(SUPERDUPER_DATA_DIR) && chmod -R 777 ${SUPERDUPER_DATA_DIR}
	@mkdir -p $(SUPERDUPER_ARTIFACTS_DIR) && chmod -R 777 ${SUPERDUPER_ARTIFACTS_DIR}
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
	SUPERDUPER_CONFIG=$(SUPERDUPER_CONFIG) pytest $(PYTEST_ARGUMENTS) ./test/unittest

usecase_testing: ## Execute usecase testing
	SUPERDUPER_CONFIG=$(SUPERDUPER_CONFIG) pytest $(PYTEST_ARGUMENTS) ./test/integration/usecase
