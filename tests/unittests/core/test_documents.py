from superduperdb.encoders.torch.tensor import tensor
from superduperdb.core.documents import Document
import torch


def test_document_encoding():
    t = tensor(torch.float, shape=(20,))

    r = Document({'x': t(torch.randn(20))})

    print(r.encode())

    Document.decode(r.encode(), encoders={'torch.float32[20]': t})
