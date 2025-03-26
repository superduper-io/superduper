DIRECTORIES ?= superduper test plugins
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
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

TEMPLATES ?= '*'

build_templates:
	if [ "$(TEMPLATES)" = "*" ]; then \
		templates_to_build=$$(ls templates/); \
	else \
		templates_to_build=$$(echo $(TEMPLATES) | tr ',' ' '); \
	fi; \
	for template in $$templates_to_build; do \
		echo $$template; \
		rm -rf templates/$$template/blobs; \
		rm -rf templates/$$template/files; \
		rm -rf templates/$$template/component.json; \
		rm -rf templates/$$template/artifact_store; \
		rm -rf templates/$$template/.ipynb_checkpoints; \
		jupyter nbconvert templates/$$template/build.ipynb --clear-output; \
	done;

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
	SUPERDUPER_CONFIG=$(SUPERDUPER_CONFIG) pytest $(PYTEST_ARGUMENTS)  test/unittest/

usecase_testing: ## Execute usecase testing
	SUPERDUPER_CONFIG=$(SUPERDUPER_CONFIG) pytest $(PYTEST_ARGUMENTS) ./test/integration/usecase
