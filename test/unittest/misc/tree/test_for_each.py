from __future__ import annotations

from pydantic import dataclasses as dc

import superduperdb as s
from superduperdb.data.tree.for_each import for_each


@dc.dataclass
class One:
    one: dict
    uno: list
    eins: str = ''


class Two(s.JSONable):
    one: One
    nine: list[One]


ONE = One(one={'a': 'b'}, uno=['hello', 'world'], eins='zed')
TWO = Two(one=ONE, nine=[ONE, One({}, [])])

BREADTH = [
    TWO,
    ONE,
    {'a': 'b'},
    'b',
    ['hello', 'world'],
    'hello',
    'world',
    'zed',
    [ONE, One(one={}, uno=[], eins='')],
    ONE,
    {'a': 'b'},
    'b',
    ['hello', 'world'],
    'hello',
    'world',
    'zed',
    One(one={}, uno=[], eins=''),
    {},
    [],
    '',
]
DEPTH = [
    ONE,
    {'a': 'b'},
    'b',
    ['hello', 'world'],
    'hello',
    'world',
    'zed',
    [ONE, One(one={}, uno=[], eins='')],
    ONE,
    {'a': 'b'},
    'b',
    ['hello', 'world'],
    'hello',
    'world',
    'zed',
    One(one={}, uno=[], eins=''),
    {},
    [],
    '',
    TWO,
]


def test_for_each_breadth():
    actual = []
    for_each(actual.append, TWO)
    assert actual == BREADTH


def test_for_each_depth():
    actual = []
    for_each(actual.append, TWO, depth_first=True)
    assert actual == DEPTH
