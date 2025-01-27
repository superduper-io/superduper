from superduper import Document
from test.utils.database import query as query_utils


def test_insert(db):
    query_utils.test_insert(db)


def test_read(db):
    query_utils.test_read(db)


def test_like(db):
    query_utils.test_like(db)


def test_insert_with_auto_schema(db):
    query_utils.test_insert_with_auto_schema(db)


def test_insert_with_diff_schemas(db):
    query_utils.test_insert_with_diff_schemas(db)


def test_auto_document_wrapping(db):
    query_utils.test_auto_document_wrapping(db)


def test_model(db):
    query_utils.test_model(db)


def test_model_query():
    query_utils.test_model_query()


def test_model_query_serialization():
    query = {
        "query": 'modela.predict("<var:abc>", documents[0], '
        'condition={"uri": "123.PDF"})',
        '_variables': {'abc': "ABC"},
        "documents": [{"a": "a"}],
        "_builds": {},
        "_files": {},
        "_path": "superduper.backends.base.query.parse_query",
    }

    decode_query = Document.decode(query)
    assert decode_query.parts[0][0] == 'predict'
    assert decode_query.parts[0][1] == ("ABC", {'a': 'a'})
    assert decode_query.parts[0][2] == {'condition': {'uri': '123.PDF'}}
