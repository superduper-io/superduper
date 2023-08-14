#!/bin/bash

set -e  # Stop execution if any command fails

python -m pip install --upgrade pip-tools pip

deps=("docs" "dev")
for dep in "${deps[@]}"
do
    echo "Running pip-compile for $dep to produce pinned dependencies"
    python -m piptools compile --extra "$dep" --upgrade --resolver backtracking -o .github/ci-pinned-requirements/"$dep".txt pyproject.toml
done
