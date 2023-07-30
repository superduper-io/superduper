import pprint

import torch
from sklearn.svm import SVC

from superduperdb.base.artifact import Artifact
from superduperdb.base.model import Model
from superduperdb.ext.torch.encoder import tensor
from superduperdb.ext.sklearn.model import Estimator


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
