PYTEST_ARGUMENTS ?=
DIRECTORIES = superduperdb test 
SUPERDUPERDB_DATA_DIR ?= .test_data

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


##@ DevKit

devkit: ## Add essential development tools
	# Add pre-commit hooks to ensure that no strange stuff are being committed.
	# https://stackoverflow.com/questions/3462955/putting-git-hooks-into-a-repository
	pip install pre-commit
	pre-commit autoupdate

	# Download tools for code quality testing
	pip install .[quality]

	# Set git to continuously update submodules
	git config --global submodule.recurse true

##@ CI Doc Functions

api-docs: ## Generate Sphinx inline-API HTML documentation
	@echo "===> Generate Sphinx HTML documentation, including API docs <==="
	rm -rf docs/api/source/
	rm -rf docs/hr/build/apidocs
	sphinx-apidoc -f -o docs/api/source superduperdb
	sphinx-build -a docs/api docs/hr/build/apidocs
	@echo "Build finished. The HTML pages are in docs/hr/build/apidocs"


hr-docs: ## Generate Docusaurus documentation and blog posts
	@echo "===> Generate docusaurus docs and blog-posts <==="
	cd docs/hr && npm i --legacy-peer-deps && npm run build
	cd ../..
	@echo "Build finished. The HTML pages are in docs/hr/build"

##@ Testing Environments

testenv_image: ## Build a sandbox image
	@echo "===> Build superduperdb/sandbox"
	docker build . -f deploy/images/superduperdb/Dockerfile -t superduperdb/sandbox --progress=plain \
		--build-arg BUILD_ENV="sandbox" \
		--build-arg SUPERDUPERDB_EXTRAS="dev"

testenv_init: ## Initialize a local Testing environment
	@echo "===> Ensure hostnames"
	@deploy/testenv/validate_hostnames.sh

	@echo "===> Ensure mongodb volume is present"
	mkdir -p deploy/testenv/$(SUPERDUPERDB_DATA_DIR)

	@echo "===> Ensure Images"
	@if docker image ls superduperdb/sandbox | grep -q "latest"; then \
        echo "superduper/sandbox found";\
        echo "*************************************************************************";\
        echo "** If Dask behaves funny, rebuild the image using 'make testenv_image' **";\
        echo "*************************************************************************";\
    else \
      	echo "superduper/sandbox not found. Please run 'make testenv_image'";\
      	exit -1;\
    fi

	SUPERDUPERDB_DATA_DIR=$(SUPERDUPERDB_DATA_DIR) docker compose -f deploy/testenv/docker-compose.yaml up --remove-orphans &

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
	# Block waiting for the testenv to become ready.
	@cd deploy/testenv/; ./wait_ready.sh

	# Run the test
	pytest $(PYTEST_ARGUMENTS) ./test/integration

fix:
	# Code formatting
	black $(DIRECTORIES)
	# Linter and code formatting
	ruff check --fix $(DIRECTORIES)
	# Linting
	mypy superduperdb

fix-and-test: ##  Lint the code before testing
	# Code formatting
	black $(DIRECTORIES)
	# Linter and code formatting
	ruff check --fix $(DIRECTORIES)
	# Linting
	mypy superduperdb
	# Unit testing
	pytest $(PYTEST_ARGUMENTS)
	# Check for missing docstrings
	interrogate superduperdb
	# Check for unused dependencies
	#deptry ./

test-and-fix: ## Test the code before linting
	# Linting
	mypy superduperdb
	# Unit testing
	pytest  $(PYTEST_ARGUMENTS)
	# Code formatting
	black $(DIRECTORIES)
	# Linter and code formatting
	ruff check --fix $(DIRECTORIES)
	# Check for missing docstrings
	interrogate superduperdb

lint-and-type-check: ##  Perform code linting and type checking
	# Linting
	mypy superduperdb
	# Code formatting
	black --check $(DIRECTORIES)
	# Linter and code formatting
	ruff check $(DIRECTORIES)
	# Check for missing docstrings
	interrogate superduperdb

test_notebooks: ## Test notebooks (argument: NOTEBOOKS=<test|dir>)
	@echo "Notebook Path: $(NOTEBOOKS)"

	@if [ -n "${NOTEBOOKS}" ]; then	\
		pytest --nbval-lax ${NOTEBOOKS}; 	\
	fi


##@ Development Sandbox Management

run_sandbox-pr: ## Run a pull request in the sandbox (argument: PR_NUMBER=555)
	@if [[ -z "${PR_NUMBER}" ]]; then echo "Usage: make run_sandbox-pr PR_NUMBER=<pull-request-number>"; exit -1; fi

	@echo "===> Checkout Pull Request #"${PR_NUMBER}" <==="

	# checkout remote repo.
	git clone --depth 1 --single-branch git@github.com:SuperDuperDB/superduperdb.git /tmp/superduperdb_pr_$(PR_NUMBER)

	# fetch specific pr
	cd /tmp/superduperdb_pr_$(PR_NUMBER) && \
	git fetch --depth 1 origin pull/$(PR_NUMBER)/head:pr_branch && \
	git checkout pr_branch

	# mount pr to sandbox
	docker run -p 8888:8888 -v /tmp/superduperdb_pr_$(PR_NUMBER):/home/superduper/pull_request superduperdb/sandbox

	# clean up the tmp directory
	rm -rf /tmp/superduperdb_pr_$(PR_NUMBER)



##@ Base Image Management

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


##@ Demo Image Management

# superduperdb/demo is a bloated image that contains everything we need to run the online demo.
build_demo: ## Build a feature-rich Docker image for demonstrations
	echo "===> build superduperdb/demo:$(RELEASE_VERSION:v%=%)"
	docker build . -f ./deploy/images/superduperdb/Dockerfile -t superduperdb/demo:$(RELEASE_VERSION:v%=%) --progress=plain --no-cache \
	--build-arg BUILD_ENV="release" \
	--build-arg SUPERDUPERDB_EXTRAS="demo"

push_demo: ## Push the superduperdb/demo:latest image
	@echo "===> release superduperdb/demo:$(RELEASE_VERSION:v%=%) <==="
	docker push superduperdb/demo:$(RELEASE_VERSION:v%=%)

	@echo "===> release superduperdb/demo:latest <==="
	docker tag superduperdb/demo:$(RELEASE_VERSION:v%=%) superduperdb/demo:latest
	docker push superduperdb/demo:latest


