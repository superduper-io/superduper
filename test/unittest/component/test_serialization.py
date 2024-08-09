from superduper.components.datatype import pickle_serializer
from superduper.components.model import ObjectModel


def test_model():
    m = ObjectModel(
        identifier='test',
        datatype=pickle_serializer,
        object=lambda x: x + 1,
    )
    m_dict = m.dict()
    assert m_dict['identifier'] == m.identifier
    assert m_dict['object'].x == m.object
    assert m_dict['datatype'].identifier == 'pickle'
