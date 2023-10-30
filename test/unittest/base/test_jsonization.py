import typing as t

import pydantic
import pytest

IS_2 = pydantic.__version__.startswith('2')

"""
This file is a pair of experiments regarding serialization of objects
containing large blobs which must be stored in a keyed cache because
they cannot be serialized.
"""


@pytest.mark.skipif(not IS_2, reason='test requires pydantic 2')
def test_jsonization1():
    class Blob(pydantic.BaseModel):
        uri: str = ''
        type_id: t.Literal['blob1'] = 'blob1'
        contents: bytes = bytes()

        @pydantic.model_serializer
        def ser_model(self) -> t.Dict[str, t.Any]:
            return {'uri': self.uri, 'type_id': self.type_id}

    blob = Blob(url='s')
    s = blob.dict()
    assert s == {'uri': '', 'type_id': 'blob1'}


@pytest.mark.skipif(not IS_2, reason='test requires pydantic 2')
def test_jsonization2():
    class Blob:
        def __init__(self, contents):
            self.contents = contents

    class Document(pydantic.BaseModel):
        uri: str = ''
        type_id: t.Literal['doc'] = 'doc'
        blob: Blob = Blob(None)

        class Config:
            arbitrary_types_allowed = True

        @pydantic.model_serializer
        def ser_model(self) -> t.Dict[str, t.Any]:
            return {'uri': self.uri, 'type_id': self.type_id}

    doc = Document()
    s = doc.dict()
    assert s == {'uri': '', 'type_id': 'doc'}


# TODO: test superduper.base.jsonable
