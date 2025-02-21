from superduper.base.datatype import dill_serializer
from superduper.components.model import ObjectModel


def test_model():
    m = ObjectModel(
        identifier='test',
        datatype='dill',
        object=lambda x: x + 1,
    )
    m_dict = m.dict()

    encoded = m_dict.encode()
    bytes = encoded['_blobs'][encoded['object'].split(':')[-1]]
    assert bytes == dill_serializer._encode_data(m.object)
