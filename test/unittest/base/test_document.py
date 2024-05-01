import dataclasses as dc
import pprint
import typing as t
from superduperdb.components.datatype import LazyArtifact
from superduperdb.components.schema import Schema
from test.db_config import DBConfig

import pytest

from superduperdb.backends.mongodb.query import Collection
from superduperdb.components.model import ObjectModel
from superduperdb.components.vector_index import vector

try:
    import torch

    from superduperdb.ext.torch.encoder import tensor
except ImportError:
    torch = None

from superduperdb.base.document import Document, _build_leaves


@pytest.fixture
def document():
    t = tensor(torch.float, shape=(20,))
    yield Document({'x': t(torch.randn(20))})


@dc.dataclass
class _db:
    datatypes: t.Dict


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_document_encoding(document):
    t = tensor(torch.float, shape=(20,))
    db = _db(datatypes={'torch.float32[20]': t})
    new_document = Document.decode(document.encode(), db)
    assert (new_document['x'].x - document['x'].x).sum() == 0


def test_flat_query_encoding():
    q = Collection('docs').find({'a': 1}).limit(2)

    r = q._deep_flat_encode(None)

    doc = Document({'x': 1})

    q = Collection('docs').like(doc, vector_index='test').find({'a': 1}).limit(2)

    r = q._deep_flat_encode(None)

    print(r)


def test_encode_decode_flattened_document():
    from superduperdb.ext.pillow.encoder import image_type
    import PIL.Image

    schema = Schema('my-schema', fields={'img': image_type(identifier='png', encodable='artifact')})

    img = PIL.Image.open('test/material/data/test.png')

    r = Document(
        {
            'x': 2,
            'img': img,
        },
        schema=schema,
    )

    encoded_r = r.deep_flat_encode()

    import yaml
    print(yaml.dump({k: v for k, v in encoded_r.items() if k != '_blobs'}))

    assert not isinstance(encoded_r, Document)
    assert isinstance(encoded_r, dict)
    assert '_leaves' in encoded_r
    assert '_blobs' in encoded_r
    assert isinstance(encoded_r['img'], str)
    assert encoded_r['img'].startswith('?artifact')
    assert isinstance(next(iter(encoded_r['_blobs'].values())), bytes)

    decoded_r = Document.deep_flat_decode(encoded_r).unpack()

    pprint.pprint(decoded_r)

    m = ObjectModel(
        'test',
        object=lambda x: x + 2,
    )

    encoded_r = m.deep_flat_encode()

    pprint.pprint(encoded_r)

    decoded_r = Document.deep_flat_decode(encoded_r)

    print(decoded_r)

    m = decoded_r.unpack()

    assert isinstance(m, ObjectModel)
    assert isinstance(m.object, LazyArtifact)

    pprint.pprint(m)

    r = m.dict()

    assert isinstance(r, Document)
    assert {'version', 'hidden', 'type_id', 'cls', 'module', 'dict'}.issubset(set(r.keys()))

    print(r)

    pprint.pprint(m.dict().deep_flat_encode())