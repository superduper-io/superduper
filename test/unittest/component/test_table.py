from superduper import Schema, Table


def test_calls_post_init():
    t = Table('test', schema=Schema('test', fields={'x': 'str'}))

    assert hasattr(t, 'version')
