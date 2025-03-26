from test.utils.database import databackend as db_utils

import pytest
from superduper import CFG
from superduper.misc.importing import load_plugin

from superduper_sql.data_backend import IbisDataBackend


@pytest.fixture
def databackend():
    plugin = load_plugin('sql')
    backend = IbisDataBackend(CFG.data_backend, plugin=plugin)
    yield backend
    backend.drop(True)


def test_list_tables_or_collections(databackend):
    db_utils.test_list_tables_or_collections(databackend)
