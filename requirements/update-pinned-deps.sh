#!/bin/bash

set -e  # Stop execution if any command fails

# Ordered by dependence - CHANGE WITH CARE!
deps=("requirements.in" "requirements-docs.in" "requirements-lint.in" "requirements-test.in" "requirements-dev.in")

for dep in "${deps[@]}"
do
    echo "Running pip-compile for $dep to produce pinned dependencies"
    pip-compile --resolver backtracking "$dep"
done