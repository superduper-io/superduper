import pytest

from superduper import Model, t
from superduper.misc.schema import Annotation, get_schema


def test_get_components():

    s, a = get_schema(Model)

    assert s['validation'] == 'componenttype'
    assert s['serve'] == 'bool'
    assert s['datatype'] == 'str'


class MyClass: ...


def test_annotation_rendering():

    a = Annotation.build(str)

    assert str(a) == "Str"

    a = Annotation.build(t.List[Model])

    assert str(a) == "List[Component]"

    a = Annotation.build(t.Dict[str, Model])

    assert str(a) == "Dict[Str, Component]"

    a = Annotation.build(t.Dict[str, t.Dict[str, Model]])

    assert str(a) == "Dict[Str, Dict[Str, Component]]"

    a = Annotation.build(t.Union[str, t.Dict[str, Model]])

    assert str(a) == "Union[Str, Dict[Str, Component]]"

    a = Annotation.build(str | t.Dict[str, Model])

    assert str(a) == "Union[Str, Dict[Str, Component]]"

    a = Annotation.build(str | t.Dict[str, Model] | t.List[Model])

    assert str(a) == "Union[Str, Dict[Str, Component], List[Component]]"

    a = Annotation.build(
        str | t.Dict[str, str] | t.List[int] | t.Tuple[t.List[str], t.Dict[str, str]]
    )

    assert (
        str(a)
        == "Union[Str, Dict[Str, Str], List[Int], Tuple[List[Str], Dict[Str, Str]]]"
    )


def test_base_types():

    a = Annotation.build(
        str | t.Dict[str, str] | t.List[int] | t.Tuple[t.List[str], t.Dict[str, str]]
    )
    assert a.base_types == {'str', 'int'}

    a = Annotation.build(
        str
        | t.Dict[str, str]
        | t.List[int]
        | t.Tuple[t.List[str], t.Dict[str, MyClass]]
    )

    assert a.base_types == {'str', 'int', 'dill'}


def test_type_mapping():

    a = Annotation.build(str)

    assert a.datatype == 'str'

    a = Annotation.build(
        str | t.Dict[str, str] | t.List[int] | t.Tuple[t.List[str], t.Dict[str, str]]
    )

    assert a.datatype == 'json'

    a = Annotation.build(
        str
        | t.Dict[str, str]
        | t.List[int]
        | t.Tuple[t.List[str], t.Dict[str, MyClass]]
    )

    assert a.datatype == 'dill'

    a = Annotation.build(
        str | t.Dict[str, str] | t.List[int] | t.Tuple[t.List[str], t.Dict[str, Model]]
    )

    with pytest.raises(AssertionError):
        _ = a.datatype

    a = Annotation.build(t.Dict[str, Model])

    assert a.datatype == 'componentdict'


class MyModelImpl(Model):
    a: str
    b: str | t.List[str] | t.Dict[str, str]
    c: t.Callable
    d: 'MyModelImpl'


def test_model_schema():

    s = MyModelImpl.class_schema

    print()
    print(s)

    assert str(s['a']) == 'str'
    assert str(s['b']) == 'JSON'
    assert str(s['c']) == 'Dill'
    assert str(s['d']) == 'ComponentType'
