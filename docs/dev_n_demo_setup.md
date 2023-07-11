

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
e.g., unix terminal, vs code terminal, or Anaconda terminal.

#### Using conda env

1. Create a python virtual environment:
`conda create -y -n superduperdb_demo python=3.10`
2. Install required packages:
`conda install -y -n superduperdb_demo pip superduperdb`
3. Activate your environment:
`conda activate superduperdb_demo`
4. If using jupyter: make the environment available in jupyter:
`python -m ipykernel install --user --name=superduperdb_demo`

#### Using python venv

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
* an editable python environment containing the `superduperdb` package,
  with core and developer dependencies
* a fork and a local `git` clone of the repository to support
  the basic GitHub flow contributor workflow we use
* completed configuration steps to enable local testing workflows


### Setting up a local mock MongoDB database server

For this prerequisite, follow the instructions of the guide
"setting up a local mock MongoDB database server" above.


### Setting up developer python environment for the `superduperdb` package



### Setting up a GitHub fork and local `git` clone of the repository



### One-time configuration for local testing workflows
