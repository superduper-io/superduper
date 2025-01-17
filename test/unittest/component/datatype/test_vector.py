import numpy


def test_auto_detect_vector(db):
    db.cfg.auto_schema = True
    db['vectors'].insert([{'x': numpy.random.randn(7)} for _ in range(3)])

    assert 'vector[7]' in db.show('datatype')

    schema = next(iter(db.show('schema')))
    schema = db.load('schema', schema)

    impl = schema.fields['x'].datatype_impl

    assert (
        impl.__module__ + '.' + impl.__class__.__name__
        == db.databackend.datatype_presets['vector']
    )
