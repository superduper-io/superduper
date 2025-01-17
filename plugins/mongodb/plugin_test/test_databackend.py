from test.utils.database import databackend as db_utils

import pytest
from superduper import CFG
from superduper.misc.importing import load_plugin

from superduper_mongodb.data_backend import MongoDBDataBackend


@pytest.fixture
def databackend():
    plugin = load_plugin('mongodb')
    backend = MongoDBDataBackend(CFG.data_backend, plugin=plugin)
    yield backend
    backend.drop(True)


@pytest.mark.skip(reason="Open this test after creating a blank table for MongoDB.")
def test_output_dest(databackend):
    db_utils.test_output_dest(databackend)


@pytest.mark.skip(reason="Open this test after creating a blank table for MongoDB.")
def test_list_tables_or_collections(databackend):
    db_utils.test_list_tables_or_collections(databackend)
