from superduper.components.model import ObjectModel


def test_auto():
    import sklearn as sklearn_native

    from superduper.ext.auto import sklearn

    m = sklearn.base.BaseEstimator(identifier='model')

    assert isinstance(m, ObjectModel)
    assert isinstance(m.object, sklearn_native.base.BaseEstimator)
    assert m.identifier == 'model'
