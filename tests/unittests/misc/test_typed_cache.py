from unittest import mock

from superduperdb.misc.typed_cache import TypedCache


def test_typed_cache():
    now = 0

    def time():
        nonlocal now
        now += 1
        return now - 1

    cache = TypedCache()

    with mock.patch('time.time', side_effect=time):
        assert cache.put(23) == 'int-0'
        assert cache.put('123') == 'str-0'

        assert cache.get('int-0') == 23
        assert cache.get('str-0') == '123'

        assert cache.put(32) == 'int-1'
        assert cache.put(235) == 'int-2'

        assert len(cache) == 4

        removed = cache.expire(3)
        assert len(cache) == 1

        expected = {int: {'0': 23, '1': 32}, str: {'0': '123'}}
        assert removed == expected
