from test.db_config import DBConfig

import pytest

from superduper.backends.mongodb.query import MongoQuery
from superduper.base.document import Document

try:
    import unstructured
except ImportError:
    unstructured = None


TEST_URL = "https://superduper.io"
TEST_FILE = "file://test/material/data/rhymes.md"


@pytest.mark.skipif(not unstructured, reason="unstructured not installed")
@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_encoder_auto(db):
    from unstructured.documents.elements import Element

    from superduper.ext.unstructured.encoder import unstructured_encoder

    db.add(unstructured_encoder)

    collection = MongoQuery(table="documents")
    to_insert = [Document({"elements": unstructured_encoder(TEST_FILE)})]
    db.execute(collection.insert_many(to_insert))

    r = db.execute(collection.find_one())
    for e in r["elements"].x:
        assert isinstance(e, Element)


@pytest.mark.skipif(not unstructured, reason="unstructured not installed")
@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_encoder_url(db):
    from unstructured.documents.elements import Element

    from superduper.ext.unstructured.encoder import unstructured_encoder

    db.add(unstructured_encoder)

    collection = MongoQuery(table="documents")
    to_insert = [Document({"elements": unstructured_encoder(TEST_URL)})]
    db.execute(collection.insert_many(to_insert))

    r = db.execute(collection.find_one())
    for e in r["elements"].x:
        assert isinstance(e, Element)
