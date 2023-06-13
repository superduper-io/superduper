from superduperdb.types.torch.tensor import tensor
from superduperdb.core.documents import Document
import torch


def test_document_encoding():
    t = tensor(torch.float)

    r = Document({'x': t(torch.randn(20))})

    print(r.encode())

    Document.decode(r.encode(), types={'torch.float32': t})
