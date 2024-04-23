import dataclasses as dc
import typing as t

import pytest
from superduperdb.ext.pillow.encoder import pil_image_hybrid_png

from test.db_config import DBConfig

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


@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty], indirect=True
)
def test_underscore_artifacts(db):
    with open('test/material/data/test.png', 'rb') as f:
        bytes = f.read()

    db.apply(pil_image_hybrid_png)

    info = db.artifact_store.save_artifact({
        'datatype': pil_image_hybrid_png.identifier,
        'bytes': bytes,
        'leaf_type': pil_image_hybrid_png.encodable
    })

    r = Document.decode({
        '_artifacts': [
            {'_content': info}
        ],
        'x': f'$artifacts/{info["file_id"]}'
    }, db=db).unpack()

    import PIL.PngImagePlugin
    assert isinstance(r['x'], PIL.PngImagePlugin.PngImageFile)


def my_function(x):
    return x + 2

@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty], indirect=True
)
def test_build_leaves(db):
    # TODO - this is how we encode data going forward

    raw = db.datatypes['dill'].encoder(my_function)
    file_id = db.artifact_store.save_artifact({'bytes': raw, 'datatype': 'dill'})['file_id']

    leaf_records = [
        {
            'leaf_type': 'artifact', 
            'cls': 'Artifact', 
            'module': 'superduperdb.components.datatype', 
            'dict': {
                'file_id': file_id,
                'datatype': 'dill',
            }
        },
        {
            'leaf_type': 'component', 
            'cls': 'ObjectModel', 
            'module': 'superduperdb.components.model', 
            'dict': {'identifier': 'test', 'object': f'_artifact/{file_id}'}
        },
        {
            'leaf_type': 'component', 
            'cls': 'Stack', 
            'module': 'superduperdb.components.stack', 
            'dict': {'identifier': 'test_stack', 'components': ['_component/model/test_']}
        },
    ]
    out = _build_leaves(leaf_records)
    print(out)