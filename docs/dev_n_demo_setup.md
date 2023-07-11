

# Getting started with SuperDuperDB - setup for demo notebooks and contributors

This page contains setup instructions for:

* new users who want to try out the SuperDuperDB python interface,
  with a mock MongoDB database and the demo notebooks
* developers who would like to contribute to the SuperDuperDB code base.

Both of the above differ from a professional deployment setup
by using a specific local mock MongoDB instance, instead of a cloud database instance.


## Setting up a local mock MongoDB database server

A local mock MongoDB instance is a prerequisite for both the user demo setup and the
basic developer-contributor setup.

For this, follow the base instructions for a
[local setup of MongoDB Community Edition](https://www.mongodb.com/docs/manual/administration/install-community/).

Note: at current, the setup is user tested on macOS and Windows,
but not on the wider variety of Linux distributions.

We encourage reports of user experiences and any setup issues
on the [superduperdb issue tracker](https://github.com/SuperDuperDB/superduperdb/issues/new/choose).


## New user setup - demo notebooks and mock MongoDB database server

Users who would like to run the demo notebooks locally need to ensure:

* a running local MongoDB server
* a python environment containing the `superduperdb` package and its core dependencies
* a local `git` clone of the repository from which the notebooks can be run


### Setting up a local mock MongoDB database server

For this prerequisite, follow the instructions of the guide
"setting up a local mock MongoDB database server" above.


### Setting up a python environment with the `superduperdb` package

To run the demo notebooks, a python environment with `superduperdb` is necessary.

There are different environment/package managers to achieve this, we outline
the workflow below for:

* `conda` with `pip`
* python `venv`

The below need to be typed into a console with python or `conda`,
e.g., unix terminal, VS Code terminal, or Anaconda terminal.

#### Using conda env

Requires: `python` and [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

1. Create a python virtual environment:
`conda create -y -n superduperdb_demo python=3.10`
2. Install required packages:
`conda install -y -n superduperdb_demo pip superduperdb`
3. Activate your environment:
`conda activate superduperdb_demo`
4. If using jupyter: make the environment available in jupyter:
`python -m ipykernel install --user --name=superduperdb_demo`

#### Using python venv

Requires: `python`

1. Create a python virtual environment:
`python -m venv superduperdb_demo`
2. Activate your environment:
`source superduperdb_demo/bin/activate`
3. Install the requirements:
`pip install superduperdb`
4. If using jupyter: make the environment available in jupyter:
`python -m ipykernel install --user --name=superduperdb_demo`


### Obtaining notebooks from a local `git` clone of the repository

To obtain a local clone of the `superduperdb` repository - including notebooks, type:

```
git clone https://github.com/superduperdb/superduperdb.git
```

in a terminal which has `git` in its path.

This will create a full git clone in the current terminal directory.

The notebooks can then be run from the `notebooks` path therein,
e.g., via `jupyter` or from the vs code graphical user interface.

Ensure to use the environment `superduperdb_demo` created above, in the notebook.


## developer-contributor setup

Users who would like to run the demo notebooks locally need to ensure:

* a running local MongoDB server
* a fork and a local `git` clone of the repository to support
  the basic GitHub flow contributor workflow we use
* an editable python environment containing the `superduperdb` package,
  with core and developer dependencies
* completed configuration steps to enable local testing workflows


### Setting up a local mock MongoDB database server

For this prerequisite, follow the instructions of the guide
"setting up a local mock MongoDB database server" above.


### Setting up a GitHub fork and local `git` clone of the repository

We follow the standard GitHub flow contribution pattern, this requires:

* a fork of `superduperdb` in your own GitHub account
* a local clone of your fork for local testing and development
* both the fork and the clone synced to the upstream

For that, you can follow [GitHub's instructions on setting up a development fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo#forking-a-repository).

Alternatively, you can use [GitHub Desktop to clone and fork](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/adding-and-cloning-repositories/cloning-and-forking-repositories-from-github-desktop)
for a more GUI-based developer experience.

Note: if you have followed the "user demo setup" instructions and
have now decided to contribute :smiley:, you will already have
a clone but it does not point to a fork.

The easiest and quickest might be fork and clone to another folder and/or
delete the original clone.
Alternatively, point your existing clone to the fork (via fiddly `git remote` varia).

### Setting up developer python environment for the `superduperdb` package

An editable python environment is required for contributing to `superduperdb`.
This can be setup with:

* python `venv`
* `conda` with `pip`
* python `venv` with `poetry`

The below need to be typed into a console, in the root directory of your
local `git` repository (see above).

#### Using python venv

Requires: `python`

1. Create a python virtual environment:
`python -m venv superduperdb_dev`
2. Activate your environment:
`source superduperdb_dev/bin/activate`
3. Install the requirements:
`pip install -e.[dev]`

#### Using conda env

Requires: `python` and [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

1. Create a python virtual environment:
`conda create -y -n superduperdb_dev python=3.10`
2. Install required packages:
`conda install -y -n superduperdb_dev pip -e .[dev]`
3. Activate your environment:
`conda activate superduperdb_dev`

#### Using poetry

Requires: `python` and [`poetry`](https://python-poetry.org/docs/#installation)

1. Create a python virtual environment:
`python -m venv superduperdb_dev`
2. Activate your environment:
`source superduperdb_dev/bin/activate`
3. Install the requirements:
`poetry install`


### One-time configuration for local testing workflows

We use `pytest` for testing and `black` and `ruff` for linting.

These packages are included in the install of the developer environment above,
but integration may have to be setup with a professional developer IDE
on project or repository basis.

You will typically need to enable `black` and `ruff` linting in developer IDEs
such as VS Code or pycharm.

For a console based or manual setup of tests, the current command to run
the full test suite locally is

```
black --check superduperdb tests && ruff check superduperdb tests && pytest
```
