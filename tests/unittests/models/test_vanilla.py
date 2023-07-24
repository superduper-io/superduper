from superduperdb.core.model import Model


def test_function_predict_one():
    function = Model(object=lambda x: x, identifier='test')
    assert function.predict(1, one=True) == 1


def test_function_predict():
    function = Model(object=lambda x: x, identifier='test')
    assert function.predict([1, 1]) == [1, 1]
