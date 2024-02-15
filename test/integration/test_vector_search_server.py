import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, database_with_default_encoders_and_model):
    from superduperdb import CFG

    vector_search = 'in_memory://localhost:8000'
    monkeypatch.setattr(CFG.cluster, 'vector_search', vector_search)

    from superduperdb.vector_search.server.app import app

    app.app.state.pool = database_with_default_encoders_and_model
    client = TestClient(app.app)

    yield client
    from superduperdb.base.config import Cluster

    monkeypatch.setattr(CFG, 'cluster', Cluster())


def test_basic_workflow(client):
    vector_index = 'test_index'

    # Add points to the vector
    # ------------------------------
    data = [{'vector': [100, 100, 100], 'id': '100'}]
    response = client.post(f"/add/search?vector_index={vector_index}", json=data)
    # Assert HTTP code
    assert response.status_code == 200
    # Assert Payload
    response = json.loads(response.content)
    assert response['message'] == 'Added vectors successfully'

    # Query points
    # ------------------------------
    response = client.post(f"/query/id/search?vector_index={vector_index}&id=100")
    by_id = json.loads(response.content)
    response = client.post(
        f"/query/search?vector_index={vector_index}", json=[100, 100, 100]
    )
    by_vector = json.loads(response.content)
    assert by_vector['ids'] == ['100']
    assert by_id['ids'] == ['100']

    # Delete points
    # ------------------------------
    response = client.post(f"/delete/search?vector_index={vector_index}", json=['100'])
    response = client.post(f"/query/id/search?vector_index={vector_index}&id=100")
    by_id = json.loads(response.content)
    assert by_id['error'] == 'KeyError'
