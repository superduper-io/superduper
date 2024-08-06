import os
from test.utils.database import metadata as metadata_utils

import pytest

from superduper import CFG
from superduper.backends.sqlalchemy.metadata import SQLAlchemyMetadata

DATABASE_URL = CFG.metadata_store or "sqlite:///:memory:"


# TODO: Remove skip when we move all the plugin tests to the new structure
skip = not os.environ.get('SUPERDUPER_CONFIG', "").endswith('ibis.yaml')

if skip:
    pytest.skip("Skipping this file for now", allow_module_level=True)


@pytest.fixture
def metadata():
    store = SQLAlchemyMetadata(DATABASE_URL)
    yield store
    store.drop(force=True)


def test_component(metadata):
    metadata_utils.test_component(metadata)


def test_parent_child(metadata):
    metadata_utils.test_parent_child(metadata)


def test_job(metadata):
    metadata_utils.test_job(metadata)
