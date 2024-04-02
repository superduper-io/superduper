## How To Contribute :rocket:


Hello! :wave: 

Thank you for considering contributing to `superduperdb`. There are many ways to contribute, and they are not limited to writing code. We welcome all contributions such as:

- bug reports
- documentation improvements
- enhancement suggestions
- expanding the tutorials and examples

This project is intended to be a community effort, and it won't be possible without your support and enthusiasm.

## Where to Start? 

If you're new to open-source development, we recommend going through the GitHub “issues” tab to find items that interest you. Once you’ve found something interesting, the next step is to create your development environment.

## Installation and setup

Once you've 'forked' and 'cloned' the code to your local machine, please follow these steps:

Get the code on your local:

```shell
# Clone and change location directory of the superduperdb repository, change the `<FORKED_NAME>` to your GitHub id
git clone git@github.com:<FORKED_NAME>/superduperdb.git
cd superduperdb
```

Set up your python environment:


```shell
# Create your Python virtual environment
python3 -m venv .venv

# Activate the Python virtual environment
. .venv/bin/activate  
```

Install the dependencies:

```shell
# Install pip-tools and latest version of pip
python3 -m pip install --upgrade pip-tools

# Install the superduperdb project in editable mode along with the developer tools
python3 -m pip install -e '.'
python3 -m pip install -r deploy/testenv/optional_requirements.txt
```

(Optional) build the docker development environment:

```shell
make testenv_image
```

## Branch workflow

We follow something called a "fork and pull request" workflow for collaborating on our project. See [here](https://gist.github.com/Chaser324/ce0505fbed06b947d962) for a great overview on what some of these mysterious terms mean! 

## Running the tests

### Unittests

These tests check that there are no basic programming errors in how 
classes and functions work internally.

```shell
make unit-testing
```

### Extension integration tests

These tests that package integrations, such as `sklearn` or `openai`
work properly.

```shell
make ext-testing
```

### Databackend integration tests

These tests check that data-backend integrations such as MongoDB or SQLite 
work as expected.

```shell
make databackend-testing
```

### Smoke tests of cluster mode

These tests check that cluster mode works as expected (`ray`, `vector-search`, `cdc`, `rest`):

```shell
make smoke-testing
```

## Getting Help 🙋

[![Slack](https://img.shields.io/badge/Slack-superduperdb-8A2BE2?logo=slack)](https://join.slack.com/t/superduperdb/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA)
[![Issues](https://img.shields.io/badge/Issues-superduperdb-8A2BE2?logo=github)](https://github.com/SuperDuperDB/superduperdb-stealth/issues)
[![Wiki](https://img.shields.io/badge/Project%20Wiki-superduperdb-8A2BE2?logo=github)](https://github.com/SuperDuperDB/superduperdb-stealth/wiki)

If you have any problems please contact a maintainer or community volunteer. The GitHub issues or the Slack channel are a great place to start. We look forward to seeing you there! :purple_heart:
