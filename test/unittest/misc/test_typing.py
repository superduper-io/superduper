from superduper.components.component import Component
from superduper.components.datatype import Dill, File
from superduper.misc import typing as t


class MyComponent(Component):
    path: t.File
    my_func: t.Blob


def new_func(x):
    return x + 1


def test_annotations(db):
    s = MyComponent.build_class_schema(db)

    assert isinstance(s.fields['path'], File)
    assert isinstance(s.fields['my_func'], Dill)

    import tempfile

    with tempfile.NamedTemporaryFile() as tmp:
        print(tmp.name)
        tmp.write('test'.encode())

        my_component = MyComponent('my_c', path=tmp.name, my_func=new_func)
        r = my_component.encode()

        assert len(r['_blobs']) == 1
        assert len(r['_files']) == 1

    import pprint

    pprint.pprint(r)
