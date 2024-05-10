import pytest
from test.db_config import DBConfig
from superduperdb import Document

@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_add_fold(db):
    breakpoint()
    new_q = db['documents'].like(Document({'text': 'some like'})).limit(n=5)
    print(new_q)

    assert str(new_q) == "documents.find({'_fold': 'valid'}).limit(5)"

