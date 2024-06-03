import dataclasses as dc
import pprint
import typing as t
from test.db_config import DBConfig

import pytest

from superduperdb.backends.mongodb.query import MongoQuery
from superduperdb.components.datatype import Artifact, DataType
from superduperdb.components.model import ObjectModel
from superduperdb.components.schema import Schema
from superduperdb.components.table import Table
from superduperdb.ext.pillow.encoder import image_type, pil_image

try:
    import torch

    from superduperdb.ext.torch.encoder import tensor
except ImportError:
    torch = None

from superduperdb.base.document import Document


@pytest.fixture
def document():
    t = tensor(dtype='float', shape=(20,))
    yield Document({'x': t(torch.randn(20))})


@dc.dataclass
class _db:
    datatypes: t.Dict


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_document_encoding(document):
    t = tensor(dtype='float', shape=(20,))
    _db(datatypes={'torch.float32[20]': t})
    new_document = Document.decode(document.encode())
    assert (new_document['x'].x - document['x'].x).sum() == 0


def test_flat_query_encoding():
    q = MongoQuery('docs').find({'a': 1}).limit(2)

    r = q._deep_flat_encode({}, {}, {})

    doc = Document({'x': 1})

    q = MongoQuery('docs').like(doc, vector_index='test').find({'a': 1}).limit(2)

    r = q._deep_flat_encode({}, {}, {})

    print(r)


def test_encode_decode_flattened_document():
    import PIL.Image

    from superduperdb.ext.pillow.encoder import image_type

    schema = Schema(
        'my-schema', fields={'img': image_type(identifier='png', encodable='artifact')}
    )

    img = PIL.Image.open('test/material/data/test.png')

    r = Document(
        {
            'x': 2,
            'img': img,
        },
        schema=schema,
    )

    encoded_r = r.encode()

    import yaml

    print(yaml.dump({k: v for k, v in encoded_r.items() if k != '_blobs'}))

    assert not isinstance(encoded_r, Document)
    assert isinstance(encoded_r, dict)
    assert '_leaves' in encoded_r
    assert '_blobs' in encoded_r
    assert encoded_r['img'].startswith('?:blob:')
    assert isinstance(next(iter(encoded_r['_blobs'].values())), bytes)


def test_encode_model():
    m = ObjectModel(
        identifier='test',
        object=lambda x: x + 2,
    )

    encoded_r = m.encode()

    pprint.pprint(encoded_r)

    decoded_r = Document.decode(encoded_r)

    print(decoded_r)

    m = decoded_r.unpack()

    assert isinstance(m, ObjectModel)
    assert isinstance(m.object, Artifact)

    pprint.pprint(m)

    r = m.dict()

    assert isinstance(r, Document)
    assert {'version', 'hidden', 'type_id', '_path'}.issubset(set(r.keys()))

    print(r)

    pprint.pprint(m.dict()._deep_flat_encode({}, {}, {}))


def test_decode_inline_data():
    import PIL.Image

    from superduperdb.ext.pillow.encoder import image_type

    it = image_type(identifier='png', encodable='artifact')

    schema = Schema('my-schema', fields={'img': it})

    img = PIL.Image.open('test/material/data/test.png')

    r = {
        'x': 2,
        'img': it.encoder(img),
    }

    r = Document.decode(r, schema=schema).unpack()
    print(r)


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_refer_to_applied_item(db):
    dt = DataType(identifier='my-type', encodable='artifact')
    db.apply(dt)

    m = ObjectModel(
        identifier='test',
        object=lambda x: x + 2,
        datatype=dt,
    )

    db.apply(m)
    r = db.metadata._get_component_by_uuid(m.uuid)
    assert r['datatype'].startswith('?:component:datatype/my-type')

    import pprint

    pprint.pprint(r)

    print(db.show('datatype'))
    dt = db.load('datatype', 'my-type', 0)
    print(dt)
    c = db.load('model', 'test')
    print(c)


@pytest.mark.parametrize("db", [DBConfig.sqldb_empty], indirect=True)
def test_column_encoding(db):
    import PIL

    img = PIL.Image.open('test/material/data/test.png')
    schema = Schema(
        'test',
        fields={
            'id': int,
            'x': int,
            'y': int,
            'img': pil_image,
        },
    )

    db.apply(Table('test', schema=schema))

    import PIL.Image

    img = PIL.Image.open('test/material/data/test.png')

    db['test'].insert(
        [
            Document({'id': 1, 'x': 1, 'y': 2, 'img': img}),
            Document({'id': 2, 'x': 3, 'y': 4, 'img': img}),
        ]
    ).execute()

    db['test'].select("x", "y", "img").execute()


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_refer_to_system(db):
    image = image_type(identifier='image', encodable='artifact')
    db.apply(image)

    import PIL.Image
    import PIL.PngImagePlugin

    img = PIL.Image.open('test/material/data/test.png')

    db.artifact_store.put_bytes(db.datatypes['image'].encoder(img), file_id='12345')

    r = {
        '_leaves': {
            'my_artifact': {
                '_path': 'superduperdb/components/datatype/LazyArtifact',
                'file_id': '12345',
                'datatype': "?db.load(datatype, image)",
            }
        },
        'img': '?my_artifact',
    }

    r = Document.decode(r, db=db).unpack()

    assert isinstance(r['img'], PIL.PngImagePlugin.PngImageFile)
