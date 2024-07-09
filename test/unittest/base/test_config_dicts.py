import io

import pytest

from superduper.base import config_dicts


def test_combine_config_dicts():
    actual = config_dicts.combine_configs(
        (
            {'one': {'two': ['three', 'four', 'five']}},
            {'one': {'three': 3}},
            {'four': None},
            {'one': {'five': {'nine': 9, 'ten': 23}}},
            {'one': {'five': {'eight': 8, 'ten': 10}}},
        )
    )

    expected = {
        'one': {
            'five': {'eight': 8, 'nine': 9, 'ten': 10},
            'three': 3,
            'two': ['three', 'four', 'five'],
        },
        'four': None,
    }

    assert expected == actual


def test_environ_dict():
    actual = config_dicts._environ_dict(
        'TEST_',
        {
            'TOAST_ONE': 'one',
            'TEST_TWO': 'two',
            'TEST_three': 'three',
        },
    )

    expected = {'two': 'two'}
    assert expected == actual


PARENT = {
    'red': {'crimson': 'Crimson', 'ruby': 'Ruby'},
    'blue': {'green': {'puce': 'Puce', 'orange': 'Orange'}, 'green_puce': 'Green Puce'},
    'blue_green': {'puce': 0, 'yellow': 30},
    'tan': {'green': {'orange': 'lime'}},
}


@pytest.mark.parametrize(
    'key, expected',
    (
        ('', []),
        ('re', []),
        ('red', [['red']]),
        ('blue_green', [['blue_green'], ['blue', 'green']]),
        ('blue_green_orange', [['blue', 'green', 'orange']]),
        (
            'blue_green_puce',
            [['blue', 'green_puce'], ['blue', 'green', 'puce'], ['blue_green', 'puce']],
        ),
    ),
)
def test_split_address(key, expected):
    actual = [list(i) for i in config_dicts._split_address(key, PARENT)]
    assert actual == expected


def test_environ_to_config_dict_many():
    environ = {
        'TEST_RED': 'red',
        'TEST_BLUE_GREEN_ORANGE': 'bge',
        'TEST_BLUE_GREEN_PUCE': 'bgp',
        'TEST_BLUE_GREEN': 'groo',
        'TEST_PURPLE': 'purple',
    }
    err = io.StringIO()
    actual = config_dicts.environ_to_config_dict('TEST_', PARENT, environ, err)
    expected = {'blue': {'green': {'orange': 'bge'}}, 'red': 'red'}

    assert actual == expected

    errors = err.getvalue().splitlines()
    assert errors == [
        'Bad environment variables:',
        'ambiguous: TEST_BLUE_GREEN_PUCE, TEST_BLUE_GREEN',
        'unknown: TEST_PURPLE',
    ]


def test_environ_to_config_dict_single():
    environ = {
        'TEST_BLUE_GREEN_ORANGE': 'bge',
    }
    err = io.StringIO()
    actual = config_dicts.environ_to_config_dict('TEST_', PARENT, environ, err)

    expected = {'blue': {'green': {'orange': 'bge'}}}
    assert actual == expected
    assert err.getvalue() == ''
