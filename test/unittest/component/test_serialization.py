import pprint

import torch
from sklearn.svm import SVC

from superduperdb.container.artifact import Artifact
from superduperdb.container.model import Model
from superduperdb.ext.sklearn.model import Estimator
from superduperdb.ext.torch.tensor import tensor


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
    pprint.pprint(empty.metadata.get_component(type_id='model', identifier='test'))
    reloaded = empty.load(type_id='model', identifier='test')
    assert isinstance(reloaded.object, Artifact)
    pprint.pprint(reloaded)
