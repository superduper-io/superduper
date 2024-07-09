from superduper.base.datalayer import ibatch


def test_ibatch():
    actual = list(ibatch(range(12), 5))
    expected = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]]
    assert actual == expected


# TODO: test superduper.components.vector_index
