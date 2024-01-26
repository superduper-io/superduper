import dataclasses as dc
import typing as t

import pytest

try:
    import torch

    from superduperdb.ext.torch.encoder import tensor
except ImportError:
    torch = None

from superduperdb.base.document import Document


@pytest.fixture
def document():
    t = tensor(torch.float, shape=(20,))
    yield Document({'x': t(torch.randn(20)), '_outputs': {'x': {'model_test': {0: 1}}}})


@dc.dataclass
class _db:
    datatypes: t.Dict


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_document_encoding(document):
    t = tensor(torch.float, shape=(20,))
    db = _db(datatypes={'torch.float32[20]': t})
    new_document = Document.decode(document.encode(), db)
    assert (new_document['x'].x - document['x'].x).sum() == 0
    assert new_document.outputs('x', 'model_test') == 1


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_document_outputs(document):
    assert document.outputs('x', 'model_test') == 1
