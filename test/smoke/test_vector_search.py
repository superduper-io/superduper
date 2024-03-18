import json

import pytest
import requests
from fastapi.testclient import TestClient

from superduperdb import CFG


@pytest.fixture
def client(database_with_default_encoders_and_model):
    from superduperdb.vector_search.server.app import app

    app.app.state.pool = database_with_default_encoders_and_model
    client = TestClient(app.app)

    yield client


def test_basic_workflow(client):
    vector_index = 'test_index'

    # Add points to the vector
    # ------------------------------
    data = [{'vector': [100, 100, 100], 'id': '100'}]
    uri = CFG.cluster.vector_search.uri
    response = requests.post(f"{uri}/add/search?vector_index={vector_index}", json=data)
    # Assert HTTP code
    assert response.status_code == 200
    # Assert Payload
    response = json.loads(response.content)
    assert response['message'] == 'Added vectors successfully'

    # Query points
    # ------------------------------
    response = requests.post(
        f"{uri}/query/id/search?vector_index={vector_index}&id=100"
    )
    by_id = json.loads(response.content)
    response = requests.post(
        f"{uri}/query/search?vector_index={vector_index}", json=[100, 100, 100]
    )
    by_vector = json.loads(response.content)
    assert by_vector['ids'] == ['100']
    assert by_id['ids'] == ['100']

    # Delete points
    # ------------------------------
    response = requests.post(
        f"{uri}/delete/search?vector_index={vector_index}", json=['100']
    )
    response = requests.post(
        f"{uri}/query/id/search?vector_index={vector_index}&id=100"
    )
    by_id = json.loads(response.content)
    assert by_id['error'] == 'KeyError'
