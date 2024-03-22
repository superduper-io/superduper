import json
import os

import pytest
import requests


@pytest.fixture
def client(monkeypatch, database_with_default_encoders_and_model):
    from superduperdb import CFG

    vector_search = os.environ['SUPERDUPERDB_CLUSTER_VECTOR_SEARCH']
    monkeypatch.setattr(CFG.cluster, 'vector_search', vector_search)

    from superduperdb.vector_search.server.app import app

    app.app.state.pool = database_with_default_encoders_and_model
    yield
    from superduperdb.base.config import Cluster

    monkeypatch.setattr(CFG, 'cluster', Cluster())


def test_basic_workflow(client):
    vector_index = 'test_index'

    # Add points to the vector
    # ------------------------------
    data = [{'vector': [100, 100, 100], 'id': '100'}]
    uri = os.environ['SUPERDUPERDB_CLUSTER_VECTOR_SEARCH']
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
