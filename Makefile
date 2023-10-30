PYTEST_ARGUMENTS ?=
DIRECTORIES = superduperdb test 

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


##@ Release Management

new_release: ## Release a new SuperDuperDB version
	@ if [[ -z "${RELEASE_VERSION}" ]]; then echo "VERSION is not set"; exit 1; fi
	@ if [[ "${RELEASE_VERSION}" == "${TAG}" ]]; then echo "no new release version. Please update VERSION file."; exit 1; fi

	@echo "** Switching to branch release-${RELEASE_VERSION}"
	@git checkout -b release-${RELEASE_VERSION}

	@echo "** Change superduperdb/__init__.py to version $(RELEASE_VERSION:v%=%)"
	@sed -ie "s/^__version__ = .*/__version__ = '$(RELEASE_VERSION:v%=%)'/" superduperdb/__init__.py
	@git add superduperdb/__init__.py

	@echo "** Change deploy/docker-compose/demo to version $(RELEASE_VERSION:v%=%)"
	sed -ie "s/superduperdb\/superduperdb:.*/superduperdb\/superduperdb:$(RELEASE_VERSION:v%=%)/" deploy/docker-compose/demo.yaml
	@git add deploy/docker-compose/demo.yaml

	@echo "** Commit Bump Version and Tags"
	@git add VERSION
	@git commit -m "Bump Version $(RELEASE_VERSION:v%=%)"
	@git tag ${RELEASE_VERSION}

	@echo "** Push release-${RELEASE_VERSION}"
	git push --set-upstream origin release-${RELEASE_VERSION} --tags


# superduperdb/superduperdb is a minimal image contains only what is needed for the framework.
build_superduperdb: ## Build minimal Docker image for general use
	echo "===> build superduperdb/superduperdb:$(RELEASE_VERSION:v%=%)"
	docker build ./deploy/images/superduperdb -t superduperdb/superduperdb:$(RELEASE_VERSION:v%=%) --progress=plain --no-cache


push_superduperdb: ## Push superduperdb/superduperdb:latest
	@echo "===> release superduperdb/superduperdb:$(RELEASE_VERSION:v%=%)"
	docker push superduperdb/superduperdb:$(RELEASE_VERSION:v%=%)

	@echo "===> release superduperdb/superduperdb:latest"
	docker tag superduperdb/superduperdb:$(RELEASE_VERSION:v%=%) superduperdb/superduperdb:latest
	docker push superduperdb/superduperdb:latest


##@ CI Doc Functions

api-docs: ## Generate Sphinx inline-API HTML documentation, including API docs
	@echo "===> Generate Sphinx HTML documentation, including API docs <==="
	rm -rf docs/api/source/
	rm -rf docs/hr/build/apidocs
	sphinx-apidoc -f -o docs/api/source superduperdb
	sphinx-build -a docs/api docs/hr/build/apidocs
	@echo "Build finished. The HTML pages are in docs/hr/build/apidocs"


hr-docs: ## Generate docusaurus and blog-posts
	@echo "===> Generate docusaurus docs and blog-posts <==="
	cd docs/hr && npm i && npm run build
	cd ../..
	@echo "Build finished. The HTML pages are in docs/hr/build"


##@ CI Testing Functions

mongo_init: ## Initialize a local MongoDB setup
	docker compose -f test/material/docker-compose.yml up mongodb mongo-init -d $(COMPOSE_ARGUMENTS)

mongo_shutdown: ## Terminate the local MongoDB setup
	docker compose -f test/material/docker-compose.yml down $(COMPOSE_ARGUMENTS)

test: mongo_init ## Perform unit testing
	pytest $(PYTEST_ARGUMENTS)

clean-test: mongo_shutdown	## Clean-up unit testing environment

fix-and-test: mongo_init ## Lint before testing
	isort $(DIRECTORIES)
	black $(DIRECTORIES)
	ruff check --fix $(DIRECTORIES)
	mypy superduperdb
	pytest $(PYTEST_ARGUMENTS)
	interrogate superduperdb

test-and-fix: mongo_init ## Test before linting.
	mypy superduperdb
	pytest $(PYTEST_ARGUMENTS)
	black $(DIRECTORIES)
	ruff check --fix $(DIRECTORIES)
	interrogate superduperdb

lint-and-type-check: ## Lint your code
	mypy superduperdb
	black --check $(DIRECTORIES)
	ruff check $(DIRECTORIES)
	interrogate superduperdb

test_notebooks: ## Test notebooks (arg: NOTEBOOKS=<test|dir>)
	@echo "Notebook Path: $(NOTEBOOKS)"

	@if [ -n "${NOTEBOOKS}" ]; then	\
		pytest --nbval ${NOTEBOOKS}; 	\
	fi


##@ DevOps Sandbox

# sandbox is a bloated image that contains everything we will need.  we don't need to expose this one to the user.
build_sandbox: ## Build bloated Docker image for development.
	@echo "===> release superduperdb/sandbox:$(RELEASE_VERSION:v%=%)"
	docker build . -f ./deploy/images/sandbox/Dockerfile -t superduperdb/sandbox:$(RELEASE_VERSION:v%=%) --progress=plain --no-cache \
	--build-arg SUPERDUPERDB_EXTRAS="testing,demo"
	#--build-arg SUPERDUPERDB_EXTRAS="torch,apis,docs,quality,testing"

run_sandbox: ## Run local repo in sandbox
	docker run -p 8888:8888 superduperdb/sandbox:$(RELEASE_VERSION:v%=%)

run_sandbox-pr: ## Run PR in sandbox (arg: PR_NUMBER=555)
	@if [[ -z "${PR_NUMBER}" ]]; then echo "Usage: make run_sandbox-pr PR_NUMBER=<pull-request-number>"; exit -1; fi

	@echo "===> Checkout Pull Request #"${PR_NUMBER}" <==="

	# checkout remote repo.
	git clone --depth 1 --single-branch git@github.com:SuperDuperDB/superduperdb.git /tmp/superduperdb_pr_$(PR_NUMBER)

	# fetch specific pr
	cd /tmp/superduperdb_pr_$(PR_NUMBER) && \
	git fetch --depth 1 origin pull/$(PR_NUMBER)/head:pr_branch && \
	git checkout pr_branch

	# mount pr to sandbox
	docker run -p 8888:8888 -v /tmp/superduperdb_pr_$(PR_NUMBER):/home/superduper/app superduperdb/sandbox:$(RELEASE_VERSION:v%=%)

	# clean up the tmp directory
	rm -rf /tmp/superduperdb_pr_$(PR_NUMBER)


##@ Demo Applications

# superduperdb/demo is a bloated image that contains everything we need to run the online demo.
build_demo: ## Build bloated Docker image for the demo
	echo "===> build superduperdb/demo:$(RELEASE_VERSION:v%=%)"
	docker build ./deploy/images/superduperdb -t superduperdb/demo:$(RELEASE_VERSION:v%=%) --progress=plain --no-cache \
	--build-arg SUPERDUPERDB_EXTRAS="torch,apis,docs,quality,testing"

push_demo: ## Push superduperdb/demo:latest
	@echo "===> release superduperdb/demo:$(RELEASE_VERSION:v%=%) <==="
	docker push superduperdb/demo:$(RELEASE_VERSION:v%=%)

	@echo "===> release superduperdb/demo:latest <==="
	docker tag superduperdb/demo:$(RELEASE_VERSION:v%=%) superduperdb/demo:latest
	docker push superduperdb/demo:latest


run_demo: ## Run SuperDuperDB demo on docker-compose
	@echo "===> Run SuperDuperDB Demo <==="

	# TODO: make it take as argument the TAG of desired image.
	docker compose -f ./deploy/docker-compose/demo.yaml up

