import json

import pytest
from fastapi.testclient import TestClient

from superduper import CFG
from superduper.base.document import Document

CFG.auto_schema = True
CFG.rest.uri = 'localhost:8000'
CFG.force_apply = True

from superduper.rest.deployed_app import app

from .mock_client import setup as _setup, teardown


@pytest.fixture
def setup():
    client = TestClient(app._app)
    yield _setup(client)
    teardown(client)


def test_health(setup):
    response = setup.get("/health")
    assert response.status_code == 200


def test_select_data(setup):
    result = setup.post('/db/execute', json={'query': 'coll.find({}, {"_id": 0})'})
    result = json.loads(result.content)
    if 'error' in result:
        raise Exception(result['messages'] + result['traceback'])
    print(result)
    assert len(result) == 2


CODE = """
from superduper import code

@code
def my_function(x):
    return x + 1
"""


def test_apply(setup):
    m = {
        '_builds': {
            'function_body': {
                '_path': 'superduper.base.code.Code',
                'code': CODE,
            },
            'my_function': {
                '_path': 'superduper.components.model.ObjectModel',
                'object': '?function_body',
                'identifier': 'my_function',
            },
        },
        '_base': '?my_function',
    }

    _ = setup.post(
        '/db/apply',
        json=m,
    )

    models = setup.get('/db/show', params={'type_id': 'model'})
    models = json.loads(models.content)

    assert models == ['my_function']


@pytest.mark.skip
def test_insert_image(setup):
    result = setup.put(
        '/db/artifact_store/put', files={"raw": ("test/material/data/test.png")}
    )
    result = json.loads(result.content)

    file_id = result['file_id']

    query = {
        '_path': 'superduper.backends.mongodb.query.parse_query',
        'query': 'coll.insert_one(documents[0])',
        '_builds': {
            'image_type': {
                '_path': 'superduper.ext.pillow.encoder.image_type',
                'encodable': 'artifact',
            },
            'my_artifact': {
                '_path': 'superduper.components.datatype.LazyArtifact',
                'blob': f'&:blob:{file_id}',
                'datatype': "?image_type",
            },
        },
        'documents': [
            {
                'img': '?my_artifact',
            }
        ],
    }

    result = setup.post(
        '/db/execute',
        json=query,
    )

    query = {
        '_path': 'superduper.backends.mongodb.query.parse_query',
        'query': 'coll.find(documents[0], documents[1])',
        'documents': [{}, {'_id': 0}],
    }

    result = setup.post(
        '/db/execute',
        json=query,
    )

    result = json.loads(result.content)
    from superduper import superduper

    db = superduper()

    result = [Document.decode(r[0], db=db).unpack() for r in result]

    assert len(result) == 3

    image_record = next(r for r in result if 'img' in r)

    from PIL.PngImagePlugin import PngImageFile

    assert isinstance(image_record['img'], PngImageFile)
