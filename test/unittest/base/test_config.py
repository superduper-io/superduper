from collections import Counter

import pydantic
import pytest

from superduperdb.base.config import Config
from superduperdb.base.jsonable import Factory, JSONable

from .test_config_dicts import PARENT

IS_2 = pydantic.__version__.startswith('2')


NAME_ERROR = """
Config.__init__() got an unexpected keyword argument \'bad_name\'
"""


def test_unknown_name():
    with pytest.raises(TypeError) as pr:
        Config(bad_name={})

    expected = NAME_ERROR.strip()
    actual = str(pr.value).strip()
    assert actual.startswith(expected)


def _dict_names(d, *address):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _dict_names(v, *address, k)
    else:
        yield '_'.join(address)


def test_dict_names():
    actual = list(_dict_names(PARENT))
    expected = [
        ('red_crimson'),
        ('red_ruby'),
        ('blue_green_puce'),
        ('blue_green_orange'),
        ('blue_green_puce'),
        ('blue_green_puce'),
        ('blue_green_yellow'),
        ('tan_green_orange'),
    ]

    assert actual == expected


def _dupes(cls):
    counts = Counter(_dict_names(cls().dict()))
    return [k for k, v in counts.items() if v > 1]


def test_config_has_no_dupes():
    assert _dupes(Config) == []


def test_find_dupes():
    class Red(JSONable):
        crimson: str = 'Crimson'
        ruby: str = 'Ruby'

    class Green(JSONable):
        puce: str = 'Puce'
        orange: str = 'Orange'

    class Blue(JSONable):
        green_puce: str = 'Green Puce'
        green: Green = Factory(Green)

    class BlueGreen(JSONable):
        puce: int = 0
        yellow: int = 30

    class Green2(JSONable):
        orange: str = 'lime'

    class Tan(JSONable):
        green: Green2 = Factory(Green2)

    class Colors(JSONable):
        red: Red = Factory(Red)
        blue: Blue = Factory(Blue)
        blue_green: BlueGreen = Factory(BlueGreen)
        tan: Tan = Factory(Tan)

    assert Colors().dict() == PARENT
    assert _dupes(Colors) == ['blue_green_puce']
