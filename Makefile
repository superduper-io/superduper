DIRECTORIES = superduperdb test
PYTEST_ARGUMENTS ?=

export SUPERDUPERDB_PYTEST_ENV_FILE ?= './deploy/testenv/users.env'

# Export variables
export SUPERDUPERDB_DATA_DIR ?= ~/.cache/superduperdb/test_data
export SUPERDUPERDB_ARTIFACTS_DIR ?= ~/.cache/superduperdb/artifacts


##@ General

# The help target prints out all targets with their descriptions organized
# beneath their categories. The categories are represented by '##@' and the
# target descriptions by '##'.
# https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_parameters
# More info on the awk command:
# http://linuxcommand.org/lc3_adv_awk.php

.DEFAULT_GOAL := help

help: ## Display this help
	@cat ./docs/api/banner.txt

	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


##@ Release Management

# RELEASE_VERSION defines the project version for the operator.
# Update this value when you upgrade the version of your project.
# The general flow is VERSION -> make new_release -> GITHUB_ACTIONS -> {make docker_push, ...}
RELEASE_VERSION=$(shell cat VERSION)

CURRENT_RELEASE=$(shell git describe --abbrev=0 --tags)

new_release: ## Release a new version of SuperDuperDB
	@ if [[ -z "${RELEASE_VERSION}" ]]; then echo "VERSION is not set"; exit 1; fi
	@ if [[ "$(RELEASE_VERSION)" == "v$(CURRENT_RELEASE)" ]]; then echo "No new release version. Please update VERSION file."; exit 1; fi

	@echo "** Switching to branch release-$(RELEASE_VERSION)"
	@git checkout -b release-$(RELEASE_VERSION)

	@echo "** Change superduperdb/__init__.py to version $(RELEASE_VERSION:v%=%)"
	@sed -ie "s/^__version__ = .*/__version__ = '$(RELEASE_VERSION:v%=%)'/" superduperdb/__init__.py
	@git add superduperdb/__init__.py

	@echo "** Commit Bump Version and Tags"
	@git add VERSION
	@git commit -m "Bump Version $(RELEASE_VERSION:v%=%)"
	@git tag $(RELEASE_VERSION)

	@echo "** Push release-$(RELEASE_VERSION)"
	git push --set-upstream origin release-$(RELEASE_VERSION) --tags

install-devkit: ## Add essential development tools
	# Add pre-commit hooks to ensure that no strange stuff are being committed.
	# https://stackoverflow.com/questions/3462955/putting-git-hooks-into-a-repository
	python -m pip install pre-commit
	#pre-commit autoupdate

	@echo "Download Docs dependencies"
	python -m pip install --user sphinx furo myst_parser

	@echo "Download Code Quality dependencies"
	python -m pip install --user black ruff mypy types-PyYAML types-requests interrogate

	@echo "Download Code Testing dependencies"
	python -m pip install --user pytest pytest-cov "nbval>=0.10.0"


##@ Code Quality

build-docs: ## Generate Docs and API
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


lint-and-type-check: ##  Perform code linting and type checking
	@echo "===> Code Formatting <==="
	black --check $(DIRECTORIES)
	ruff check $(DIRECTORIES)

	@echo "===> Static Typing Check <==="
	mypy superduperdb
	# Check for missing docstrings
	interrogate superduperdb
	# Check for unused dependencies
	# deptry ./
	# Check for deadcode
	# vulture ./

fix-and-test: ##  Lint the code before testing
	# Code formatting
	black $(DIRECTORIES)
	# Linter and code formatting
	ruff check --fix $(DIRECTORIES)
	# Linting
	rm -rf .mypy_cache/
	mypy superduperdb


##@ Image Management

# superduperdb/superduperdb is a minimal image contains only what is needed for the framework.
build_superduperdb: ## Build a minimal Docker image for general use
	echo "===> build superduperdb/superduperdb:$(RELEASE_VERSION:v%=%)"
	docker build . -f ./deploy/images/superduperdb/Dockerfile -t superduperdb/superduperdb:$(RELEASE_VERSION:v%=%) --progress=plain --no-cache \
	--build-arg BUILD_ENV="release"


push_superduperdb: ## Push the superduperdb/superduperdb:latest image
	@echo "===> release superduperdb/superduperdb:$(RELEASE_VERSION:v%=%)"
	docker push superduperdb/superduperdb:$(RELEASE_VERSION:v%=%)

	@echo "===> release superduperdb/superduperdb:latest"
	docker tag superduperdb/superduperdb:$(RELEASE_VERSION:v%=%) superduperdb/superduperdb:latest
	docker push superduperdb/superduperdb:latest



testenv_image: ## Build a sandbox image
	@echo "===> Build superduperdb/sandbox"
	docker build . -f deploy/images/superduperdb/Dockerfile -t superduperdb/sandbox --progress=plain \
		--build-arg BUILD_ENV="sandbox" \
		--build-arg SUPERDUPERDB_EXTRAS="dev" \


##@ Testing Environment

testenv_init: ## Initialize a local Testing environment
	@echo "===> Discover Images"
	@if docker image ls superduperdb/sandbox | grep -q "latest"; then \
        echo "superduper/sandbox found";\
    else \
      	echo "superduper/sandbox not found. Please run 'make testenv_image'";\
      	exit -1;\
    fi

	@echo "===> Discover Hostnames"
	@deploy/testenv/validate_hostnames.sh

	@echo "===> Discover Paths"
	echo "SUPERDUPERDB_DATA_DIR: $(SUPERDUPERDB_DATA_DIR)"
	echo "SUPERDUPERDB_ARTIFACTS_DIR: $(SUPERDUPERDB_ARTIFACTS_DIR)"

	@mkdir -p $(SUPERDUPERDB_DATA_DIR) && chmod -R 777 ${SUPERDUPERDB_DATA_DIR}
	@mkdir -p $(SUPERDUPERDB_ARTIFACTS_DIR) && chmod -R 777 ${SUPERDUPERDB_ARTIFACTS_DIR}


	@echo "===> Run TestEnv"
	docker compose -f deploy/testenv/docker-compose.yaml up --remove-orphans &

	@echo "===> Waiting for TestEnv to become ready"
	@cd deploy/testenv/; ./wait_ready.sh

testenv_shutdown: ## Terminate the local Testing environment
	@echo "===> Shutting down the local Testing environment"
	docker compose -f deploy/testenv/docker-compose.yaml down


testenv_restart: testenv_shutdown testenv_init ## Restart the local Testing environment

testdb_init: ## Initialize databases in Docker
	cd deploy/databases/; docker compose up --remove-orphans &

testdb_shutdown: ## Terminate Databases Containers
	cd deploy/databases/; docker compose down


##@ CI Testing Functions

unit-testing: ## Execute unit testing
	pytest $(PYTEST_ARGUMENTS) ./test/unittest/

integration-testing: ## Execute integration testing
	pytest $(PYTEST_ARGUMENTS) ./test/integration

test_notebooks: ## Test notebooks (argument: NOTEBOOKS=<test|dir>)
	@echo "Notebook Path: $(NOTEBOOKS)"

	@if [ -n "$(NOTEBOOKS)" ]; then	\
		pytest --nbval-lax $(NOTEBOOKS); 	\
	fi
