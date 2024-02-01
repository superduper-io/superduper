## How To Contribute :rocket:


Hello! :wave: 

Thank you for considering contributing to `superduperdb`. There are many ways to contribute, and they are not limited to writing code. We welcome all contributions such as:

- bug reports
- documentation improvements
- enhancement suggestions
- expanding the tutorials and examples

This project is intended to be a community effort, and it won't be possible without your support and enthusiasm.

## Where to Start? :boom:
If you're new to open-source development, we recommend going through the GitHub “issues” tab to find items that interest you. Once you’ve found something interesting, the next step is to create your development environment.

We follow something called a "fork and pull request" workflow for collaborating on our project. See [here](https://gist.github.com/Chaser324/ce0505fbed06b947d962) for a great overview on what some of these mysterious terms mean! 

**Before getting started , make sure you have a `Python 3.11` version**

Once you've 'forked' and 'cloned' the code to your local machine, please follow these steps:

```shell
# Clone and change location directory of the superduperdb repository, change the `<FORKED_NAME>` to your GitHub id
git clone git@github.com:<FORKED_NAME>/superduperdb.git
cd superduperdb

# Create your Python virtual environment
python3 -m venv .venv

# Activate the Python virtual environment
. .venv/bin/activate  
```

```shell
# Install pip-tools and latest version of pip
python3 -m pip install --upgrade pip-tools

# Install the superduperdb project in editable mode along with the developer tools
python3 -m pip install -e '.[dev]'
```

The final steps to creating a development environment involve installing [MongoDB](https://www.mongodb.com/docs/manual/installation/), [Docker](https://docs.docker.com/engine/install/) and [pandoc](https://pandoc.org/installing.html). Once you get this far, you are all set to start contributing - ship it! :shipit:

## Interacting with the CI system :cold_sweat:

### How do I update the `requirements` files for CI? :link:

1. Activate your [developer](#where-to-start--boom-) environment.
2. Update the Changelog to reflect the changes you did.
3. Create a Pull Request.

### Why do we have so many `requirements` files for CI? :confused:

The short answer is so that we can create reproducible environments for our continuous integration suite. We use `pip-tools` to create a pinned `.txt` version of dependencies that satisfies our version range constraints (for those familiar with `poetry`, this is similar to `poetry.lock`). A long-form answer of all this is available [here](https://hynek.me/articles/semver-will-not-save-you/).

## Getting Help 🙋


[![Slack](https://img.shields.io/badge/Slack-superduperdb-8A2BE2?logo=slack)](https://join.slack.com/t/superduperdb/shared_invite/zt-1zuojj0k0-RjAYBs1TDsvEa7yaFGa6QA)
[![Issues](https://img.shields.io/badge/Issues-superduperdb-8A2BE2?logo=github)](https://github.com/SuperDuperDB/superduperdb-stealth/issues)
[![Wiki](https://img.shields.io/badge/Project%20Wiki-superduperdb-8A2BE2?logo=github)](https://github.com/SuperDuperDB/superduperdb-stealth/wiki)

If you have any problems please contact a maintainer or community volunteer. The GitHub issues or the Slack channel are a great place to start. We look forward to seeing you there! :purple_heart:
