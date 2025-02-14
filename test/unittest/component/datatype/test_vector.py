import numpy


def test_auto_detect_vector(db):
    db.cfg.auto_schema = True
    db['vectors'].insert([{'x': numpy.random.randn(7)} for _ in range(3)])

    t = db.load('Table', 'vectors')

    assert t.fields['x'] == 'vector[float64:7]'
