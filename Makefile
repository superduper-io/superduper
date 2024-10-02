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

install_all_plugins:
	@plugins=""; \
	for plugin in $$(ls plugins); do \
		if [ "$$plugin" != "template" -a -d "plugins/$$plugin" -a -f "plugins/$$plugin/pyproject.toml" ]; then \
			plugins="$$plugins $$plugin"; \
		fi \
	done; \
	echo "Found plugins:$$plugins"; \
	for plugin in $$plugins; do \
		echo "Installing $$plugin..."; \
		python -m pip install -e "plugins/$$plugin[test]"; \
	done
%:
	@:

fix-and-check: ##  Lint the code before testing
	# Code formatting
	black $(DIRECTORIES)
	# Linter and code formatting
	ruff check --fix $(DIRECTORIES)
	# Linting

	@if [ -d .mypy_cache ]; then rm -rf .mypy_cache; fi
	mypy superduper

##@ CI Testing Functions

rest_testing: ## Execute rest unit tests
	SUPERDUPER_CONFIG=$(SUPERDUPER_CONFIG) pytest $(PYTEST_ARGUMENTS) ./test/rest

unit_testing: ## Execute unit testing
	SUPERDUPER_CONFIG=$(SUPERDUPER_CONFIG) pytest $(PYTEST_ARGUMENTS) ./test/unittest

usecase_testing: ## Execute usecase testing
	SUPERDUPER_CONFIG=$(SUPERDUPER_CONFIG) pytest $(PYTEST_ARGUMENTS) ./test/integration/usecase
