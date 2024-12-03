from test.utils.database import artifact_store as artifact_store_utils

import pytest
from superduper import CFG

from superduper_mongodb.data_backend import MongoDBDataBackend

DATABASE_URL = CFG.artifact_store or CFG.data_backend or ""


if DATABASE_URL.split(":")[0] not in ["mongodb", "mongodb+srv"]:
    pytest.skip("MongoDB is not configured, skipping tests...", allow_module_level=True)


@pytest.fixture
def artifact_store():
    artifact_store = MongoDBDataBackend(DATABASE_URL).build_artifact_store()
    yield artifact_store
    artifact_store.drop(True)


def test_bytes(artifact_store):
    artifact_store_utils.test_bytes(artifact_store)


def test_file(artifact_store):
    artifact_store_utils.test_file(artifact_store)
