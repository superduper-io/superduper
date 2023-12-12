import pytest
import torch

from superduperdb.components.model import Model
from superduperdb.components.serializer import serializers
from superduperdb.components.stack import Stack


@pytest.fixture
def stack():
    pickler = serializers['pickle']
    bytes_ = pickler.encode(torch.nn.Linear(10, 32))

    return {
        'identifier': 'my_stack',
        'artifacts': [
            {
                'serializer': 'pickle',
                'bytes': bytes_,
            },
        ],
        'components': [
            {'module': 'superduperdb.ext.pillow', 'variable': 'pil_image'},
            {
                'module': 'superduperdb.components.model',
                'cls': 'Model',
                'dict': {
                    'identifier': 'my_model',
                    'object': '$artifacts[0]',
                    'encoder': '$components.pil_image',
                },
            },
        ],
    }


def test_from_dict(stack):
    s = Stack.from_dict(stack)
    assert isinstance(s.components[1], Model)
