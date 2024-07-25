from test.utils.database import databackend as db_utils

import pytest
from superduper import CFG
from superduper.backends.ibis.data_backend import IbisDataBackend


@pytest.fixture
def databackend():
    backend = IbisDataBackend(CFG.data_backend)
    yield backend
    backend.drop(True)


def test_output_dest(databackend):
    db_utils.test_output_dest(databackend)


def test_query_builder(databackend):
    db_utils.test_query_builder(databackend)


def test_list_tables_or_collections(databackend):
    db_utils.test_list_tables_or_collections(databackend)
