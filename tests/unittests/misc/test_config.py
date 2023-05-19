from . test_dicts import PARENT
from collections import Counter
from pathlib import Path
from pydantic import ValidationError
from superduperdb.misc.config import _Factory, _Model, Config
import copy
import json
import pytest

DEFAULT_CONFIG_FILE = Path(__file__).parents[3] / 'default-configs.json'

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

def test_default_config():
    # If this test fails, try running
    #
    #    $ python -m tests.unittests.misc.test_config
    #
    # to rebuild default-configs.json
    assert Config().dict() == json.loads(DEFAULT_CONFIG_FILE.read_text())


def write_default_config():
    DEFAULT_CONFIG_FILE.write_text(json.dumps(Config().dict(), indent=2))


def test_copy_config():
    cf = Config(**DATA)
    assert cf.ray.deployments[0].database == 'mnist'

    assert cf.dict() == DATA


def test_type_error():
    d2 = copy.deepcopy(DATA)
    d2['dask']['port'] = 'bad port'

    with pytest.raises(ValidationError) as pr:
        Config(**d2)
    assert str(pr.value).strip() == TYPE_ERROR.strip()


def test_unknown_name():
    with pytest.raises(ValidationError) as pr:
        c = Config(bad_name={}, **DATA)
    assert str(pr.value).strip() == NAME_ERROR.strip()


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
        ('tan_green_orange')
    ]

    assert actual == expected


def _dupes(cls):
    counts = Counter(_dict_names(cls().dict()))
    return [k for k, v in counts.items() if v > 1]


def test_config_has_no_dupes():
    assert _dupes(Config) == []


def test_find_dupes():
    class Red(_Model):
        crimson = 'Crimson'
        ruby = 'Ruby'

    class Green(_Model):
        puce = 'Puce'
        orange = 'Orange'

    class Blue(_Model):
        green_puce = 'Green Puce'
        green: Green = _Factory(Green)

    class BlueGreen(_Model):
        puce = 0
        yellow = 30

    class Green2(_Model):
        orange = 'lime'

    class Tan(_Model):
        green: Green2 = _Factory(Green2)

    class Colors(_Model):
        red: Red = _Factory(Red)
        blue: Blue = _Factory(Blue)
        blue_green: BlueGreen = _Factory(BlueGreen)
        tan: Tan = _Factory(Tan)

    assert Colors().dict() == PARENT
    assert _dupes(Colors) == ['blue_green_puce']


DATA = {
     'remote': False,
     'vector_search': {
          'host': 'localhost',
          'port': 5001
     },
     'dask': {
          'ip': 'localhost',
          'port': 8786,
          'serializers': ['pickle', 'dill'],
          'deserializers': ['pickle', 'dill']
     },
     'ray': {
          'deployments': [
               {
                    'database': 'mnist',
                    'model': 'lenet'
               }
          ],
          'host': '127.0.0.1',
          'port': 8000
     },
     'model_server': {
          'host': 'localhost',
          'port': 5002
     },
     'mongodb': {
          'host': 'localhost',
          'port': 27017
     },
     'notebook': {
          'token': '...',
          'port': 8888,
          'ip': '0.0.0.0'
     },
     'apis': {
          'openai': {
               'api_key': 'sk-...'
          }
     }
}


if __name__ == '__main__':
    write_default_config()
    print('Wrote', DEFAULT_CONFIG_FILE)
