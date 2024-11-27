import typing as t

import pytest

from superduper import Component, Schema, Table
from superduper.components.datatype import (
    File,
    dill_serializer,
    file,
    pickle_encoder,
    pickle_serializer,
)
from superduper.misc.typing import JSON, Blob


class TestComponent(Component):
    _fields = {'a': dill_serializer, 'b': file}

    a: t.Callable
    b: str | None = None


class TestComponentWithAnnotation(Component):
    a: Blob
    b: JSON


class TestUnannotatedComponent(Component):
    a: t.Callable
    b: t.Optional[t.Callable]


def test_annotation_without_fields():
    assert TestComponentWithAnnotation._fields['a'] == 'default'
    assert TestComponentWithAnnotation._fields['b'] == 'json'
    c = TestComponentWithAnnotation('test', a=lambda x: x, b={'x': 2})

    print(c.dict())


def test_schema_with_bytes_encoding(db):
    db.apply(
        Table(
            'documents',
            schema=Schema('_schema/documents', fields={'txt': pickle_encoder}),
        )
    )

    t = db.load('table', 'documents')

    assert t.schema.db is not None

    db.databackend.bytes_encoding = 'base64'

    db['documents'].insert([{'txt': 'testing 123'}]).execute()

    try:
        r = db.databackend.db['documents'].find_one()
    except Exception:
        return

    print(r)

    assert isinstance(r['txt'], str)

    r = db['documents'].find_one()


def test_schema_with_blobs(db):
    db.apply(
        Table(
            'documents',
            schema=Schema('_schema/documents', fields={'txt': pickle_serializer}),
        )
    )

    db['documents'].insert([{'txt': 'testing 123'}]).execute()

    r = db['documents'].select().tolist()[0]

    from superduper.components.datatype import Blob

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
            schema=Schema('_schema/documents', fields={'my_file': file}),
        )
    )
    db['documents'].insert([{'my_file': tmp_file}]).execute()

    # only the references are loaded when data is selected
    r = db['documents'].select().tolist()[0]

    # loaded document contains a pointer to the file
    assert isinstance(r['my_file'], File)

    # however the path has not been populated
    assert not r['my_file'].path

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

    from superduper.components.datatype import Blob

    assert isinstance(r['a'], Blob)

    assert r_encoded['a'].startswith('&:blob:')
    assert r_encoded['b'].startswith('&:file:')


def test_auto_infer_fields():
    s = TestUnannotatedComponent.build_class_schema()

    assert isinstance(s, Schema)

    import pprint

    pprint.pprint(s)

    assert list(s.fields.keys()) == ['a', 'b']


def test_wrap_function_with_blob():
    r = TestComponent('test', a=lambda x: x + 1).dict()
    from superduper.components.datatype import Blob

    assert isinstance(r['a'], Blob)
