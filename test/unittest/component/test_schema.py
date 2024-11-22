from superduper import Schema, Table
from superduper.components.datatype import pickle_encoder


def test_schema_with_bytes_encoding(db):
    db.apply(
        Table(
            'documents',
            schema=Schema('_schema/documents', fields={'txt': pickle_encoder}),
        )
    )

    t = db.load('table', 'documents')

    assert t.schema.db is not None

    db.databackend.bytes_encoding = 'base64'

    db['documents'].insert([{'txt': 'testing 123'}]).execute()

    try:
        r = db.databackend.db['documents'].find_one()
    except Exception:
        return

    print(r)

    assert isinstance(r['txt'], str)

    r = db['documents'].find_one()
