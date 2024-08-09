import pprint

import numpy as np

from superduper.backends.base.query import Query
from superduper.base.constant import KEY_BLOBS, KEY_BUILDS
from superduper.base.document import Document
from superduper.components.datatype import (
    Artifact,
    DataType,
    pickle_serializer,
)
from superduper.components.model import ObjectModel
from superduper.components.schema import Schema
from superduper.components.table import Table


def test_document_encoding():
    document = Document({'x': pickle_serializer(np.random.rand(20))})
    new_document = Document.decode(
        document.encode(), getters={'component': lambda x: pickle_serializer}
    )
    assert (new_document['x'].x - document['x'].x).sum() == 0


def test_flat_query_encoding():
    q = Query(table='docs').find({'a': 1}).limit(2)

    r = q._deep_flat_encode({}, {}, {})

    doc = Document({'x': 1})

    q = Query(table='docs').like(doc, vector_index='test').find({'a': 1}).limit(2)

    r = q._deep_flat_encode({}, {}, {})

    print(r)


def test_encode_decode_flattened_document():
    data = np.array([1, 2, 3])
    from superduper.components.datatype import pickle_serializer

    schema = Schema('my-schema', fields={'data': pickle_serializer})

    r = Document(
        {
            'x': 2,
            'data': data,
        },
        schema=schema,
    )

    encoded_r = r.encode()

    import yaml

    print(yaml.dump({k: v for k, v in encoded_r.items() if k != KEY_BLOBS}))

    assert not isinstance(encoded_r, Document)
    assert isinstance(encoded_r, dict)
    assert KEY_BUILDS in encoded_r
    assert KEY_BLOBS in encoded_r
    assert encoded_r['data'].startswith('&:blob:')
    assert isinstance(next(iter(encoded_r[KEY_BLOBS].values())), bytes)


def test_encode_model():
    m = ObjectModel(
        identifier='test',
        object=lambda x: x + 2,
    )

    encoded_r = m.encode()

    pprint.pprint(encoded_r)

    decoded_r = Document.decode(
        encoded_r, getters={'blob': lambda x: encoded_r[KEY_BLOBS][x]}
    )

    print(decoded_r)

    m = decoded_r.unpack()

    assert isinstance(m, ObjectModel)
    assert isinstance(m.object, Artifact)

    pprint.pprint(m)

    r = m.dict()

    assert isinstance(r, Document)
    assert {'version', 'hidden', 'type_id', '_path'}.issubset(set(r.keys()))

    print(r)

    pprint.pprint(m.dict().encode())


def test_decode_inline_data():
    schema = Schema('my-schema', fields={'data': pickle_serializer})

    r = {
        'x': 2,
        'data': pickle_serializer.encode_data(np.random.randn(20)),
    }

    r = Document.decode(r, schema=schema).unpack()
    print(r)


def test_refer_to_applied_item(db):
    dt = DataType(identifier='my-type', encodable='artifact')
    db.apply(dt)

    m = ObjectModel(
        identifier='test',
        object=lambda x: x + 2,
        datatype=dt,
    )

    db.apply(m)
    r = db.metadata.get_component_by_uuid(m.uuid)

    assert r['datatype'].startswith('&:component:datatype:my-type')

    import pprint

    pprint.pprint(r)

    print(db.show('datatype'))
    dt = db.load('datatype', 'my-type', 0)
    print(dt)
    c = db.load('model', 'test')
    print(c)


def test_column_encoding(db):
    schema = Schema(
        'test',
        fields={
            'id': int,
            'x': int,
            'y': int,
            'data': pickle_serializer,
        },
    )

    db.apply(Table('test', schema=schema))
    data = np.random.rand(20)
    db['test'].insert(
        [
            Document({'id': 1, 'x': 1, 'y': 2, 'data': data}),
            Document({'id': 2, 'x': 3, 'y': 4, 'data': data}),
        ]
    ).execute()

    db['test'].select().execute()


def test_refer_to_system(db):
    db.apply(pickle_serializer)

    db.artifact_store.put_bytes(
        pickle_serializer.encode_data(np.random.rand(3)), file_id='12345'
    )

    r = {
        '_builds': {
            'my_artifact': {
                '_path': 'superduper.components.datatype.LazyArtifact',
                'blob': '&:blob:12345',
                'datatype': "&:component:datatype:pickle",
            }
        },
        'data': '?my_artifact',
    }

    r = Document.decode(r, db=db).unpack()

    assert isinstance(r['data'], np.ndarray)
