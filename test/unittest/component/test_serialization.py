import pytest

try:
    import torch

    from superduper.ext.torch.encoder import tensor
except ImportError:
    torch = None

from sklearn.svm import SVC

from superduper.components.model import ObjectModel
from superduper.ext.sklearn.model import Estimator


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_model():
    m = ObjectModel(
        identifier='test',
        datatype=tensor(dtype='float', shape=(32,)),
        object=torch.nn.Linear(13, 18),
    )
    m_dict = m.dict()
    assert m_dict['identifier'] == m.identifier
    assert m_dict['object'].x == m.object
    assert m_dict['datatype'].identifier == 'torch-float32[32]'


def test_sklearn(db):
    m = Estimator(
        identifier='test',
        object=SVC(),
        datatype=tensor(dtype='float', shape=(32,)),
    )
    assert 'object' in m.artifact_schema.fields
    db.apply(m)
    assert db.show('model') == ['test']
    assert db.show('datatype') == []

    reloaded = db.load(type_id='model', identifier='test')
    reloaded.init()
    assert isinstance(reloaded.object, SVC)
