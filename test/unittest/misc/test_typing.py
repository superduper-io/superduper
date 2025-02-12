from superduper.components.component import Component
from superduper.misc import typing as st


class MyComponent(Component):
    path: st.File
    my_func: st.Dill


def new_func(x):
    return x + 1


def test_annotations(db):

    assert MyComponent._new_fields['path'] == 'file'
    assert MyComponent._new_fields['my_func'] == 'dill'

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
