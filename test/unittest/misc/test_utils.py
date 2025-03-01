import typing as t

from superduper import Component


class MyComponent(Component):
    a: int
    b: str
    c: t.Dict


def test_hash_item():

    c1 = MyComponent('test', a=1, b='test-1', c={'this': 'is a test'})

    c2 = MyComponent('test', a=1, b='test-1', c={'this': 'is a test'})

    assert c1.uuid == c2.uuid

    c3 = MyComponent('test', a=1, b='test-2', c={'this': 'is a test'})

    assert c1.hash != c3.hash

    assert c1.get_merkle_tree(False)['b'] != c3.get_merkle_tree(False)['b']

    for k in c1.get_merkle_tree(False):
        if k == 'b':
            continue
        assert (
            c1.get_merkle_tree(False)[k] == c3.get_merkle_tree(False)[k]
        ), 'Trees differed at key: {}'.format(k)
