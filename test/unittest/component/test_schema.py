import typing as t

import pytest

from superduper import Component, Schema, Table
from superduper.base.datatype import (
    Blob,
    FileItem,
    dill_serializer,
    file,
)
from superduper.misc import typing as st


class TestComponent(Component):
    _fields = {'a': dill_serializer, 'b': file}

    a: t.Callable
    b: st.File


class TestUnannotatedComponent(Component):
    a: t.Callable
    b: t.Optional[t.Callable]


def test_schema_with_bytes_encoding(db):
    db.apply(
        Table(
            'documents',
            fields={'txt': 'pickleencoder'},
        )
    )

    db['documents'].insert([{'txt': 'testing 123'}])

    r = db.databackend.db['documents'].get()

    print(r)

    assert isinstance(r['txt'], str)

    r = db['documents'].get()


def test_schema_with_blobs(db):
    db.apply(
        Table(
            'documents',
            fields={'txt': 'pickle'},
        )
    )

    db['documents'].insert([{'txt': 'testing 123'}])

    r = db['documents'].get()

    assert isinstance(r['txt'], Blob)

    # artifacts are loaded lazily and initially empty
    assert r['txt'].bytes is None

    # artifacts are downloaded and decoded with `.unpack()`
    assert r.unpack()['txt'] == 'testing 123'


@pytest.fixture
def tmp_file():
    file = '/tmp/test_schema_with_file.txt'
    with open(file, 'a') as f:
        f.write('Hello 123')
        pass

    yield file

    import os

    os.remove(file)


def test_schema_with_file(db, tmp_file):
    # the `file` is a datatype which copies a file
    # to the artifact store when a reference document
    # containing a file field is inserted
    db.apply(
        Table(
            'documents',
            fields={'my_file': 'file'},
        )
    )
    db['documents'].insert([{'my_file': tmp_file}])

    # only the references are loaded when data is selected
    r = db['documents'].get()

    # loaded document contains a pointer to the file
    assert isinstance(r['my_file'], FileItem)

    # unpacking the document copies the file to the artifact-store
    rr = r.unpack()

    # the path has been populated
    assert r['my_file'].path

    # and is also now local
    import os

    assert os.path.exists(r['my_file'].path)

    # the unpacked value contains the local path
    # this may be different from the original file path
    assert rr['my_file'] == r['my_file'].path

    with open(rr['my_file']) as f:
        f.read().split('\n')[0] = 'Hello 123'


def test_component_serializes_with_schema(db, tmp_file):
    c = TestComponent('test', a='testing testing 123', b=tmp_file)

    r = c.dict()

    r_encoded = r.encode()

    import pprint

    pprint.pprint(r.schema)

    pprint.pprint(r_encoded)

    assert r_encoded['a'].startswith('&:blob:')
    assert r_encoded['b'].startswith('&:file:')


def test_auto_infer_fields(db):
    s = TestUnannotatedComponent.class_schema

    assert isinstance(s, Schema)

    import pprint

    pprint.pprint(s)

    assert set(s.fields.keys()).issuperset({'upstream', 'a', 'b'})
