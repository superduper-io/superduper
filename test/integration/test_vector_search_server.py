import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(database_with_default_encoders_and_model):
    from superduperdb.vector_search.server.app import app

    app.app.state.pool = database_with_default_encoders_and_model
    client = TestClient(app.app)

    yield client


def test_basic_workflow(client):
    vector_index = 'test_index'
    data = [{'vector': [100, 100, 100], 'id': '100'}]
    response = client.post(
        f"/vector_search/add/search?vector_index={vector_index}", json=data
    )
    assert response.status_code == 200
    response = json.loads(response.content)
    assert response['message'] == 'Added vectors successfuly'

    response = client.post(
        f"/vector_search/query/id/search?vector_index={vector_index}&id=100"
    )
    by_id = json.loads(response.content)
    response = client.post(
        f"/vector_search/query/search?vector_index={vector_index}", json=[100, 100, 100]
    )
    by_vector = json.loads(response.content)
    assert by_vector['ids'] == ['100']
    assert by_id['ids'] == ['100']

    response = client.post(
        f"/vector_search/delete/search?vector_index={vector_index}", json=['100']
    )
    response = client.post(
        f"/vector_search/query/id/search?vector_index={vector_index}&id=100"
    )
    by_id = json.loads(response.content)
    assert by_id['error'] == 'KeyError'
