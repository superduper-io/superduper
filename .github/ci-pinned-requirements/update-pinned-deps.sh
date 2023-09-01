#!/bin/bash

set -e  # Stop execution if any command fails

python -m pip install --upgrade pip-tools pip

echo "Running pip-compile for core to produce pinned dependencies"
python -m piptools compile \
  -o .github/ci-pinned-requirements/core.txt \
  --resolver backtracking \
  --upgrade \
  pyproject.toml

deps=("docs" "dev")
for dep in "${deps[@]}"
do
    echo "Running pip-compile for $dep to produce pinned dependencies"
    python -m piptools compile \
    -o .github/ci-pinned-requirements/"$dep".txt \
    --extra "$dep" \
    --resolver backtracking \
    --upgrade \
    pyproject.toml
done
