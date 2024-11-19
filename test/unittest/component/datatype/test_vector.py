import numpy
import pprint


def test_auto_detect_vector(db):
    db.cfg.auto_schema = True
    db['vectors'].insert([{'x': numpy.random.randn(7)} for _ in range(3)]).execute()
    assert 'vector[7]' in db.show('datatype')