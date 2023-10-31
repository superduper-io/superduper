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

    yield Document({'x': t(torch.randn(20)), '_outputs': {'x': {'model_test': 1}}})


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_document_encoding(document):
    t = tensor(torch.float, shape=(20,))
    print(document.encode())
    Document.decode(document.encode(), encoders={'torch.float32[20]': t})


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_document_outputs(document):
    assert document.outputs('x', 'model_test') == 1


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_only_uri(local_data_layer):
    r = Document(
        Document.decode(
            {'x': {'_content': {'uri': 'foo', 'encoder': 'torch.float32[8]'}}},
            encoders=local_data_layer.encoders,
        )
    )
    assert r['x'].uri == 'foo'
