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
