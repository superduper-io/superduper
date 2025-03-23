import os
import pprint
import tempfile

import numpy as np
import pytest

from superduper.base.constant import KEY_BLOBS, KEY_BUILDS
from superduper.base.datatype import (
    pickle_encoder,
    pickle_serializer,
)
from superduper.base.document import Document
from superduper.base.schema import Schema
from superduper.components.component import Component
from superduper.components.listener import Listener
from superduper.components.model import ObjectModel
from superduper.components.table import Table


def test_document_encoding(db):
    schema = Schema(fields={'x': pickle_serializer})
    document = Document({'x': np.random.rand(20)}, schema=schema)
    new_document = Document.decode(
        document.encode(),
        schema=schema,
    )
    new_document = new_document.unpack()
    assert (new_document['x'] - document['x']).sum() == 0


def test_flat_query_encoding(db):
    # TODO what is being tested here??

    t = db['docs']

    q = t.filter(t['a'] == 1).limit(2)

    r = q.encode()

    q = t.like({'x': 1}, vector_index='test').filter(t['a'] == 1).limit(2)

    r = q.encode()

    print(r)


def test_encode_decode_flattened_document():
    data = np.array([1, 2, 3])
    from superduper.base.datatype import pickle_serializer

    schema = Schema({'data': pickle_serializer})

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


@pytest.mark.skip
def test_encode_model_with_remote_file(db):
    r = {
        '_base': '?20d76167d4a6ad7fe00250e8359d0dca',
        '_builds': {
            'file': {
                '_path': 'superduper.components.datatype.get_serializer',
                'method': 'file',
                'encodable': 'file',
                'type_id': 'datatype',
                'version': None,
                'uuid': '1ef3aaae626a45aa836b82f493acc874',
            },
            '20d76167d4a6ad7fe00250e8359d0dca': {
                '_path': 'superduper.components.datatype.File',
                'uuid': '43d651a7990241f3accbd4b67b77b069',
                'datatype': '?file',
                'uri': None,
                'x': '&:file:file://./README.md',
            },
        },
        '_blobs': {},
        '_files': {},
    }

    r = Document.decode(r, db=db).unpack()
    assert os.path.exists(r)
    with open(r, 'rb') as r:
        read = r.readlines()

    with open('./README.md', 'rb') as r:
        assert r.readlines() == read


@pytest.mark.skip
def test_encode_model_with_remote_blob():
    m = ObjectModel(
        identifier='test',
        object=lambda x: x + 2,
    )

    encoded_r = m.encode()
    with tempfile.TemporaryDirectory() as temp_dir:
        blob_key = [
            k for k in encoded_r['_builds'] if k not in ('test', 'datatype:dill_lazy')
        ][0]

        blob = encoded_r['_blobs'][blob_key]
        temp_file_path = os.path.join(temp_dir, blob_key)
        with open(temp_file_path, 'wb') as f:
            f.write(blob)

        encoded_r['_builds'][blob_key]['blob'] = f'&:blob:file://{temp_file_path}'
        decoded_r = Document.decode(encoded_r)

    m = decoded_r.unpack()
    m.object.init()
    assert m.object.x(1) == 3


def test_encode_model(db):
    m = ObjectModel(
        identifier='test',
        object=lambda x: x + 2,
    )

    encoded_r = m.encode()

    pprint.pprint(encoded_r)

    decoded_r = ObjectModel.decode(encoded_r)

    print(decoded_r)

    m = decoded_r.setup()

    assert isinstance(m, ObjectModel)
    assert callable(m.object)

    r = m.dict()

    assert isinstance(r, Document)
    assert {'version', 'status', '_path'}.issubset(set(r.keys()))

    print(r)

    pprint.pprint(m.dict().encode())


def test_decode_inline_data(db):
    schema = Schema({'data': 'pickleencoder'})

    r = {
        'x': 2,
        'data': pickle_encoder.encode_data(np.random.randn(20), None),
    }

    r = Document.decode(r, schema=schema).unpack()
    print(r)


def test_column_encoding(db):
    fields = {
        'id': 'str',
        'x': 'int',
        'y': 'int',
        'data': 'pickle',
    }

    db.apply(Table('test', fields=fields))
    data = np.random.rand(20)
    db['test'].insert(
        [
            {'id': '1', 'x': 1, 'y': 2, 'data': data},
            {'id': '2', 'x': 3, 'y': 4, 'data': data},
        ]
    )

    db['test'].select().execute()


def test_refer_to_system(db):
    db.artifact_store.put_bytes(
        pickle_serializer._encode_data(np.random.rand(3)), file_id='12345'
    )

    r = {
        'data': '&:blob:12345',
    }

    r = Document.decode(r, db=db, schema=Schema({'data': pickle_serializer})).unpack()

    assert isinstance(r['data'], np.ndarray)


def test_encode_same_identifier():
    model = ObjectModel(identifier="a", object=lambda x: x, datatype='str')
    listener = Listener(model=model, identifier="a", key="a", select=None)

    encode_data = listener.encode()
    listener = Component.decode(encode_data)

    assert listener.identifier == "a"
    assert listener.model.identifier == "a"


def test_diff():
    r1 = Document({'a': 1, 'b': 2})

    r2 = Document({'a': 1, 'b': 3})

    diff = r1.diff(r2)

    assert set(diff.keys()) == {'b'}
