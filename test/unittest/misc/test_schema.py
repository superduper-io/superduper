from superduper import Model
from superduper.misc.schema import get_schema


def test_get_components():

    s, a = get_schema(Model)

    assert s['validation'] == 'componenttype'
    assert s['serve'] == 'bool'
    assert s['datatype'] == 'str'
