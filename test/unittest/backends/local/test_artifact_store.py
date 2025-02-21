from test.utils.database import artifact_store as artifact_store_utils

import pytest

from superduper.base.artifacts import FileSystemArtifactStore


@pytest.fixture
def artifact_store(tmpdir):
    artifact_store = FileSystemArtifactStore(str(tmpdir))
    yield artifact_store
    artifact_store.drop(True)


def test_bytes(artifact_store: FileSystemArtifactStore):
    artifact_store_utils.test_bytes(artifact_store)


def test_file(artifact_store: FileSystemArtifactStore):
    artifact_store_utils.test_file(artifact_store)
