import json
import os
import pytest

from .mock_client import curl_post, setup as _setup, teardown
from superduperdb import CFG


@pytest.fixture
def setup():
    yield _setup()
    teardown()


def test_select_data(setup):
    form = {
        "documents": [],
        "query": [
            "documents.find()"
        ],
        "artifacts": [],
    }
    result = curl_post('/db/execute', form)
    print(result)
    assert len(result) == 2


def test_insert_image(setup):
    request = f"""curl -X 'PUT' \
        '{CFG.cluster.rest.uri}/db/artifact_store/save_artifact?datatype=image' \
        -H 'accept: application/json' \
        -H 'Content-Type: multipart/form-data' \
        -s \
        -F 'raw=@test/material/data/test.png;type=image/png'"""

    result = os.popen(request).read()
    result = json.loads(result)
    assert 'file_id' in result
    file_id = result['file_id']

    form = {
        "documents": [
            {
                "img": {
                    "_content": {
                        "file_id": result["file_id"], 
                        "datatype": "image",
                        "leaf_type": "lazy_artifact",
                        "uri": None,
                    }
                }
            },
        ],
        "query": [
            f"documents.insert_one($documents[0])"
        ],
    }
    form = json.dumps(form)

    request = f"""curl -X 'POST' \
        '{CFG.cluster.rest.uri}/db/execute' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -s \
        -d '{form}'"""

    print('making request')
    result = json.loads(os.popen(request).read())
    if 'error' in result:
        raise Exception(result['messages'])
    print(result)

    form = json.dumps({
        "documents": [],
        "query": [
            "documents.find()"
        ],
    })

    request = f"""curl -X 'POST' \
        '{CFG.cluster.rest.uri}/db/execute' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -s \
        -d '{form}'"""

    print('making request')
    result = os.popen(request).read()
    print(result)
    result = json.loads(result)
    result = next(r for r in result if 'img' in r)
    assert result['img']['_content']['file_id'] == file_id