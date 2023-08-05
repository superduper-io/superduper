import pytest
import torch

from superduperdb.container.document import Document
from superduperdb.ext.torch.tensor import tensor


@pytest.fixture(scope='function')
def document():
    t = tensor(torch.float, shape=(20,))

    yield Document({'x': t(torch.randn(20)), '_outputs': {'x': {'model_test': 1}}})


def test_document_encoding(document):
    t = tensor(torch.float, shape=(20,))
    print(document.encode())
    Document.decode(document.encode(), encoders={'torch.float32[20]': t})


def test_document_outputs(document):
    assert document.outputs('x', 'model_test') == 1
