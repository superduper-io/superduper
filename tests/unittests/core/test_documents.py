import pytest

from superduperdb.encoders.torch.tensor import tensor
from superduperdb.core.document import Document
import torch


@pytest.fixture(scope='function')
def document():
    t = tensor(torch.float, shape=(20,))

    r = Document({'x': t(torch.randn(20)), '_outputs': {'x': {'model_test': 1}}})
    yield r


def test_document_encoding(document):
    t = tensor(torch.float, shape=(20,))
    print(document.encode())
    Document.decode(document.encode(), encoders={'torch.float32[20]': t})


def test_document_outputs(document):
    assert document.outputs('x', 'model_test') == 1
