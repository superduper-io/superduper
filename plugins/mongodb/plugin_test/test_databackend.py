from test.utils.database import databackend as db_utils

import pytest
from superduper import CFG

from superduper_mongodb.data_backend import MongoDataBackend


@pytest.fixture
def databackend():
    backend = MongoDataBackend(CFG.data_backend)
    yield backend
    backend.drop(True)


@pytest.mark.skip(reason="Open this test after creating a blank table for MongoDB.")
def test_output_dest(databackend):
    db_utils.test_output_dest(databackend)


def test_query_builder(databackend):
    db_utils.test_query_builder(databackend)


@pytest.mark.skip(reason="Open this test after creating a blank table for MongoDB.")
def test_list_tables_or_collections(databackend):
    db_utils.test_list_tables_or_collections(databackend)
