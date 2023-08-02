from collections import Counter

import pydantic
import pytest

from superduperdb.base.config import Config, Factory, JSONable, Notebook

from .test_config_dicts import PARENT

IS_2 = pydantic.__version__.startswith('2')

TYPE_ERROR = """
1 validation error for Config
dask -> port
  value is not a valid integer (type=type_error.integer)
"""
NAME_ERROR = """
1 validation error for Config
bad_name
  extra fields not permitted (type=value_error.extra)
"""
VALIDATION_ERROR = """
1 validation error for Notebook
__root__
  At most one of password and token may be set (type=value_error)
"""

TYPE_ERROR2 = """
1 validation error for Config
dask.port
  Input should be a valid integer, unable to parse string as an integer\
 [type=int_parsing, input_value='bad port', input_type=str]
"""
NAME_ERROR2 = """
1 validation error for Config
bad_name
  Extra inputs are not permitted [type=extra_forbidden, input_value={}, input_type=dict]
"""
VALIDATION_ERROR2 = """
1 validation error for Notebook
  Value error, At most one of password and token may be set [type=value_error,\
 input_value={'password': 'password', 'token': 'token'}, input_type=dict]
"""


def test_type_error():
    d2 = Config().dict()
    d2['dask']['port'] = 'bad port'

    with pytest.raises(pydantic.ValidationError) as pr:
        Config(**d2)

    expected = (TYPE_ERROR2 if IS_2 else TYPE_ERROR).strip()
    actual = str(pr.value).strip()
    assert actual.startswith(expected)


def test_unknown_name():
    with pytest.raises(pydantic.ValidationError) as pr:
        Config(bad_name={})

    expected = (NAME_ERROR2 if IS_2 else NAME_ERROR).strip()
    actual = str(pr.value).strip()
    assert actual.startswith(expected)


def test_validation():
    with pytest.raises(pydantic.ValidationError) as pr:
        Notebook(password='password', token='token')

    expected = (VALIDATION_ERROR2 if IS_2 else VALIDATION_ERROR).strip()
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
