from test.utils.database import query as query_utils

from superduper.backends.base.query import parse_query


def test_insert(db):
    query_utils.test_insert(db)


def test_atomic_parse(db):
    query_utils.test_atomic_parse(db)


def test_encode_decode_data(db):
    query_utils.test_encode_decode_data(db)


def test_filter_select(db):
    query_utils.test_filter_select(db)


def test_filter(db):
    query_utils.test_filter(db)


def test_select_one_col(db):
    query_utils.test_select_one_col(db)


def test_select_all_cols(db):
    query_utils.test_select_all_cols(db)


def test_select_table(db):
    query_utils.test_select_table(db)


def test_ids(db):
    query_utils.test_ids(db)


def test_subset(db):
    query_utils.test_subset(db)


def test_outputs(db):
    query_utils.test_outputs(db)


def test_read(db):
    query_utils.test_read(db)


def test_like(db):
    query_utils.test_like(db)


def test_insert_with_auto_schema(db):
    query_utils.test_insert_with_auto_schema(db)


def test_insert_with_diff_schemas(db):
    query_utils.test_insert_with_diff_schemas(db)


def test_parse_outputs_query(db):
    q = parse_query(
        query='_outputs__listener1__9bc4a01366f24603.select()',
        documents=[],
        db=db,
    )

    assert len(q) == 2
