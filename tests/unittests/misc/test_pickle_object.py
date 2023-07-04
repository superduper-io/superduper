from superduperdb.misc.pickle_object import PickleObject


def test_pickle_object():
    obj = {'a': [1230, 92, None, False], 'b': test_pickle_object}

    po = PickleObject(obj)

    d = po.dict()
    assert list(d) == ['pickled']
    assert isinstance(d['pickled'], bytes)

    po2 = PickleObject(**d)

    assert po.object is obj is po.object
    assert po2.object == obj
    assert po2.object is not obj
