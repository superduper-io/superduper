# FAQ :question:

## How do I add/remove a dependency? :link:

0. Are you **really sure** that you need to do this? :innocent:
1. Make sure you have read step 0. :smiling_imp: 
2. Activate your developer environment (see [this](#installing-a-developer-environment-💻) section if you are unsure how to do this).
3. Add the dependency to the relevant *.in* file(s). For example, a core dependency belongs in `requirements.in`, a test dependency belongs in `requirements-test.in`. Please try to provide as wide a range of compatible dependency versions as possible.
4. Update the pinned dependencies (the *.txt* files required for our CI) by running `./update-pinned-deps.sh`

## Why do we have so many `requirements` files? :confused:

The short answer is so that we can create reproducible environments for our continuous integration suite. Each file with a suffix `.in` provides the acceptable version ranges for our dependencies in a given environment. For example `requirements-test.in` provides the dependency version ranges for our test suite. We then use `pip-tools` to create a pinned `.txt` version of dependencies that satisfies our version range constraints (for those familiar with `poetry`, this is similar to `poetry.lock`). A long-form answer of all this is available [here](https://hynek.me/articles/semver-will-not-save-you/).

## How can we handle optional features (like support for a new vector database) in our project? :crystal_ball:

[PEP 621](https://peps.python.org/pep-0621/) means that a Python project's core metadata can now be written entirely in the `pyproject.toml` file. This PEP provides an `optional-dependencies` feature for installing a project with optional features. For the user, this looks like `python -m pip install mypackage[extra-cool-feature]`. To tackle issue #384, this will likely be the correct place to store the relevant metadata.