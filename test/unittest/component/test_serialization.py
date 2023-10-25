import pprint

import pytest

try:
    import torch

    from superduperdb.ext.torch.encoder import tensor
except ImportError:
    torch = None
from sklearn.svm import SVC

from superduperdb.base.artifact import Artifact
from superduperdb.components.model import Model
from superduperdb.ext.sklearn.model import Estimator


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_model():
    m = Model(
        identifier='test',
        encoder=tensor(torch.float, shape=(32,)),
        object=torch.nn.Linear(13, 18),
    )
    print(m)
    print(m.dict())


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_sklearn(empty):
    m = Estimator(
        identifier='test',
        object=SVC(),
        encoder=tensor(torch.float, shape=(32,)),
    )
    empty.add(m)
    assert empty.metadata.component_collection.count_documents({}) == 2
    pprint.pprint(empty.metadata.get_component(type_id='model', identifier='test'))
    reloaded = empty.load(type_id='model', identifier='test')
    assert isinstance(reloaded.object, Artifact)
    pprint.pprint(reloaded)
