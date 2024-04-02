import base64
import json
import os
import pytest

from superduperdb import superduper
from superduperdb.base.document import Document
from superduperdb.components.component import Component
from superduperdb.rest.utils import _parse_query_part


def insert(data):
    data = {
        "documents": data,
        "query": [
            f"documents.insert_many($documents)"
        ],
        "artifacts": [],
    }

    data = json.dumps(data)

    print(data)

    request = f"""curl -X 'POST' \
        'http://localhost:8002/db/execute' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -s \
        -d '{data}'"""

    print(request)

    result = os.popen(request).read()

    print(result)


def apply(component):
    data = json.dumps({'component': {component['dict']['identifier']: component}})
    print(data)
    request = f"""curl -X 'POST' \
        'http://localhost:8002/db/apply' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{data}'"""
    result = os.popen(request).read()
    print(result)


def delete():
    data = {
        "documents": [],
        "query": [
            "documents.delete_many({})"
        ],
        "artifacts": [],
    }

    data = json.dumps(data)

    print(data)

    request = f"""curl -X 'POST' \
        'http://localhost:8002/db/execute' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -s \
        -d '{data}'"""

    print(request)

    result = os.popen(request).read()

    print(result)


@pytest.fixture
def setup():
    data = [
        {"x": [1, 2, 3, 4, 5], "y": 'test'},
        {"x": [6, 7, 8, 9, 10], "y": 'test'},
    ]
    insert(data)
    apply({
        'cls': 'image_type',
        'module': 'superduperdb.ext.pillow.encoder',
        'dict': {
            'identifier': 'image',
            'media_type': 'image/png'
        }
    })
    yield
    delete()


def test_select_data(setup):
    form = {
        "documents": [],
        "query": [
            "documents.find()"
        ],
        "artifacts": [],
    }
    form = json.dumps(form)

    request = f"""curl -X 'POST' \
        'http://localhost:8002/db/execute' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -s \
        -d '{form}'"""

    result = json.loads(os.popen(request).read())
    print(result)

    assert len(result) == 2


def test_insert_image(setup):

    request = f"""curl -X 'PUT' \
        'http://localhost:8002/db/artifact_store/save_artifact?datatype=image' \
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
                "img":{
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
        'http://localhost:8002/db/execute' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -s \
        -d '{form}'"""

    print('making request')
    result = os.popen(request).read()
    print(result)

    form = json.dumps({
        "documents": [],
        "query": [
            "documents.find()"
        ],
    })

    request = f"""curl -X 'POST' \
        'http://localhost:8002/db/execute' \
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