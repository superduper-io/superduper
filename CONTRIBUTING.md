## How To Contribute :rocket:


Hello! :wave: 

Thank you for considering contributing to `superduper`. There are many ways to contribute, and they are not limited to writing code. We welcome all contributions such as:

- bug reports
- documentation improvements
- enhancement suggestions
- expanding the reusable-snippets and use-cases

This project is intended to be a community effort, and it won't be possible without your support and enthusiasm.

## Where to Start? 

If you're new to open-source development, we recommend going through the [GitHub “issues” tab](https://github.com/superduper-io/superduper/issues) to find items that interest you. Once you’ve found something interesting, the next step is to create your development environment.

## Installation and setup

Once you've 'forked' and 'cloned' the code to your local machine, please follow these steps:

Get the code on your local:

```shell
# Clone and change location directory of the superduper repository, change the `<FORKED_NAME>` to your GitHub id
git clone git@github.com:<FORKED_NAME>/superduper.git
cd superduper
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

# Install the superduper project in editable mode along with the developer tools
python3 -m pip install -e '.'
python3 -m pip install -r deploy/installations/testenv_requirements.txt
make install_devkit
```

(Optional) build the docker development environment:

```shell
make build_sandbox
```

## Branch workflow

We follow something called a "fork and pull request" workflow for collaborating on our project. See [here](https://gist.github.com/Chaser324/ce0505fbed06b947d962) for a great overview on what some of these mysterious terms mean! 

## Running the tests

### Unittests

These tests check that there are no basic programming errors in how 
classes and functions work internally.

```shell
make unit_testing
```

### Extension integration tests

These tests that package integrations, such as `sklearn` or `openai`
work properly.

```shell
make ext_testing
```

### Databackend integration tests

These tests check that data-backend integrations such as MongoDB or SQLite 
work as expected.

```shell
make databackend_testing
```

### Smoke tests of cluster mode

These tests check that cluster mode works as expected (`ray`, `vector-search`, `cdc`, `rest`):

```shell
make smoke_testing
```

## Create an issue

If you have an unsolvable problem or find a bug with the code, we
would love it if you could create a useful [issue on GitHub](https://github.com/superduper-io/superduper-stealth/issues).

Creating a useful issue, is itself a useful skill. Think about following these pointers:

- Add the "bug label" to flag the issue as a bug
- Make sure the issue contains the ***minimal code*** needed to create the issue:
  - Remove padding code, unnecessary setup etc. 
  - Make it as easy as possible to recreate the problem.
- Always include the traceback in the issue
- To flag the issue to the team, escalate this in the Slack channels
- Tag relevant people who have worked on that issue, or know the feature

## Getting Help 🙋

[![Slack](https://img.shields.io/badge/Slack-superduperdb-8A2BE2?logo=slack)](https://join.slack.com/t/superduperdb/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA)
[![Issues](https://img.shields.io/badge/Issues-superduperdb-8A2BE2?logo=github)](https://github.com/superduper-io/superduper-stealth/issues)
[![Wiki](https://img.shields.io/badge/Project%20Wiki-superduperdb-8A2BE2?logo=github)](https://github.com/superduper-io/superduper-stealth/wiki)

If you have any problems please contact a maintainer or community volunteer. The GitHub issues or the Slack channel are a great place to start. We look forward to seeing you there! :purple_heart:
