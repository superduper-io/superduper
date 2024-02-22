import pytest

try:
    import torch

    from superduperdb.ext.torch.encoder import tensor
except ImportError:
    torch = None
from test.db_config import DBConfig

from sklearn.svm import SVC

from superduperdb.components.model import ObjectModel
from superduperdb.ext.sklearn.model import Estimator


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_model():
    m = ObjectModel(
        identifier='test',
        datatype=tensor(torch.float, shape=(32,)),
        object=torch.nn.Linear(13, 18),
    )
    m_dict = m.dict()['dict']
    assert m_dict['identifier'] == m.identifier
    assert m_dict['object'].x == m.object
    assert m_dict['datatype'].identifier == 'torch.float32[32]'


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    "db", [DBConfig.mongodb_empty, DBConfig.sqldb_empty], indirect=True
)
def test_sklearn(db):
    m = Estimator(
        identifier='test',
        object=SVC(),
        datatype=tensor(torch.float, shape=(32,)),
    )
    assert 'object' in m.artifact_schema.fields

    db.add(m)
    assert db.show('model') == ['test']
    assert db.show('datatype') == ['torch.float32[32]']

    reloaded = db.load(type_id='model', identifier='test')
    reloaded.init()
    assert isinstance(reloaded.object, SVC)
