import json

import pytest
from fastapi.testclient import TestClient

from superduper import CFG

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
    result = setup.post('/db/execute', json={'query': 'coll.select()'})
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
                '_path': 'superduper.components.model.ImportedModel',
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
