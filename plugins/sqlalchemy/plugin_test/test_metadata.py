from test.utils.database import metadata as metadata_utils

import pytest
from superduper import CFG, model, superduper

from superduper_sqlalchemy.metadata import SQLAlchemyMetadata

DATABASE_URL = CFG.metadata_store or "sqlite://"


@pytest.fixture
def metadata():
    store = SQLAlchemyMetadata(DATABASE_URL)
    store._batched = False
    yield store
    store.drop(force=True)


def test_component(metadata):
    metadata_utils.test_component(metadata)


def test_parent_child(metadata):
    metadata_utils.test_parent_child(metadata)


def test_job(metadata):
    metadata_utils.test_job(metadata)


def test_artifact_relation(metadata):
    metadata_utils.test_artifact_relation(metadata)


def test_cleanup_metadata():

    db = superduper(DATABASE_URL)

    @model
    def test(x): return x + 1

    db.apply(test, force=True)

    assert 'test' in db.show('model'), 'The model was not added to metadata'

    db.remove('model', 'test', force=True)

    assert not db.show(), 'The metadata was not cleared up'

    assert not db.metadata._cache, f'Cache not cleared: {db.metadata._cache}'

