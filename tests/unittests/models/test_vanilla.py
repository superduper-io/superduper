from superduperdb.models.vanilla.wrapper import Function


def test_function_predict_one():
    function = Function(object=lambda x: x, identifier='test')
    assert function.predict(1) == 1


def test_function_predict():
    function = Function(object=lambda x: x, identifier='test')
    assert function.predict([1, 1]) == [1, 1]
