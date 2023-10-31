#!/bin/bash

set -eu

if [[ "$(python --version)" != "Python 3.8"* ]] ; then
    echo 'This script must be run under Python 3.8'
    exit 1
fi

python -m pip install --upgrade pip-tools pip

echo "Running pip-compile for core to produce pinned dependencies"
python -m piptools compile \
  -o .github/ci-pinned-requirements/core.txt \
  --resolver backtracking \
  --upgrade \
  pyproject.toml

deps=("docs" "dev" "demo")
for dep in "${deps[@]}"
do
    echo "Running pip-compile for $dep to produce pinned dependencies"
    python -m piptools compile \
    -o .github/ci-pinned-requirements/"$dep".txt \
    --extra "$dep" \
    --resolver backtracking \
    --upgrade \
    pyproject.toml

    sed -i '/^superduperdb\[/d' .github/ci-pinned-requirements/"$dep".txt
done

