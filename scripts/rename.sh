#!/bin/bash

set -eux

make-dir() {
    for dir in "$@"
    do
        mkdir -p $dir
        touch $dir/__init__.py
        git add $dir/__init__.py
    done
}

find . --name __pycache__ | xargs rm -Rf

#
# Fix tests
#

rm -Rf test
git mv tests test
git mv test/unittests test/unittest

pushd test/unittest

rm -Rf container db model server
git mv models model
git mv datalayer db

git mv training/test_query_dataset.py db/mongodb
git mv queries/mongodb/test_pymongo.py db/mongodb
git mv fetchers/test_downloaders.py db/base

git rm -f fetchers/__init__.py queries/__init__.py queries/mongodb/__init__.py

make-dir \
    base \
    misc/cache

git mv misc/test_config*.py base
git mv misc/test_*cache.py misc/cache

popd
pushd superduperdb

make-dir \
    base \
    data \
    data/cache \
    data/tree

rm -Rf container db model server

git mv cluster server
git mv core container
git mv datalayer db
git mv models model

git mv \
    misc/config.py \
    misc/configs.py \
    misc/config_dicts.py \
    misc/jsonable.py \
    misc/logger.py \
    base/

git mv encoders data/encoder
git mv misc/serialization.py data/encoder/
git mv misc/downloads.py db/base/
git mv misc/*cache.py data/cache/

# Not used
git rm metrics/vector_search.py metrics/__init__.py
git mv metrics/classification.py model/
