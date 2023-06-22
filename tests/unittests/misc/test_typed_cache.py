from superduperdb.misc.typed_cache import TypedCache


def test_typed_cache():
    cache = TypedCache()

    assert cache.put(23) == 'int-0'
    assert cache.put('123') == 'str-0'

    assert cache.get('int-0') == 23
    assert cache.get('str-0') == '123'
