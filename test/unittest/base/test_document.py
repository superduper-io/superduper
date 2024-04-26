import dataclasses as dc
import pprint
import typing as t
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


def my_function(x):
    return x + 2


def test_encode_decode_flattened(db):
    m = ObjectModel(
        identifier='test',
        object=lambda x: x,
        datatype=vector(384),
    )

    r, bytes = m.export(format='json')
    pprint.pprint(r)

    for file_id in bytes:
        db.artifact_store.save_artifact({'bytes': bytes[file_id], 'datatype': 'dill'})

    r['_leaves'] = _build_leaves(r['_leaves'], db)[0]
    pprint.pprint(r['_leaves'][r['_base']])


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_build_leaves(db):
    raw = db.datatypes['dill'].encoder(my_function)
    file_id = db.artifact_store.save_artifact({'bytes': raw, 'datatype': 'dill'})[
        'file_id'
    ]
    leaf_records = [
        {
            'leaf_type': 'artifact',
            'cls': 'Artifact',
            'module': 'superduperdb.components.datatype',
            'dict': {
                'file_id': file_id,
                'datatype': 'dill',
            },
        },
        {
            'leaf_type': 'component',
            'cls': 'ObjectModel',
            'module': 'superduperdb.components.model',
            'dict': {'identifier': 'test', 'object': f'_artifact/{file_id}'},
        },
        {
            'leaf_type': 'component',
            'cls': 'Stack',
            'module': 'superduperdb.components.stack',
            'dict': {
                'identifier': 'test_stack',
                'components': ['_component/model/test'],
            },
        },
    ]
    out = _build_leaves(leaf_records)
    print(out)


def test_flat_query_encoding():
    q = Collection('docs').find({'a': 1}).limit(2)

    r = q._deep_flat_encode(None)

    doc = Document({'x': 1})

    q = Collection('docs').like(doc, vector_index='test').find({'a': 1}).limit(2)

    r = q._deep_flat_encode(None)

    print(r)
