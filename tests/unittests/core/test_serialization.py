import pprint

from superduperdb.core.base import Artifact
from superduperdb.encoders.torch.tensor import tensor
from superduperdb.core.model import Model
import torch

from superduperdb.models.sklearn.wrapper import Estimator
from sklearn.svm import SVC


def test_model():
    m = Model(
        identifier='test',
        encoder=tensor(torch.float, shape=(32,)),
        object=torch.nn.Linear(13, 18),
    )
    print(m)
    print(m.dict())


def test_sklearn(empty):
    m = Estimator(
        identifier='test',
        object=SVC(),
        encoder=tensor(torch.float, shape=(32,)),
    )
    empty.add(m)
    assert empty.metadata.object_collection.count_documents({}) == 2
    pprint.pprint(empty.metadata.get_component(variety='model', identifier='test'))
    reloaded = empty.load(variety='model', identifier='test')
    assert isinstance(reloaded.object, Artifact)
    pprint.pprint(reloaded)
