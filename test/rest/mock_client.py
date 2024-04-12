import json
import os


def curl_get(endpoint, data):
    raise NotImplementedError


def curl_post(endpoint, data):
    data = json.dumps(data)
    request = f"""curl -X 'POST' \
        'http://localhost:8002{endpoint}' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -s \
        -d '{data}'"""
    print('CURL REQUEST:')
    print(request)
    result = os.popen(request).read()
    assert result, f'POST request to {endpoint} with {data} returned empty response'
    result = json.loads(result), f'Response is not valid JSON: {result}'
    return result


def curl_put(endpoint, file, media_type, params=None):
    if params is not None:
        params = '?' + '&'.join([f'{k}={v}' for k, v in params.items()])
    request = f"""curl -X 'PUT' \
        'http://localhost:8002{endpoint}{params}' \
        -H 'accept: application/json' \
        -H 'Content-Type: multipart/form-data' \
        -s \
        -F 'raw=@{file};type={media_type}'"""
    print('CURL REQUEST:')
    print(request)
    result = os.popen(request).read()
    assert result, f'PUT request to {endpoint} with {params} and {file} returned empty response'
    return result
    

def insert(data):
    data = {
        "documents": data,
        "query": [
            f"documents.insert_many($documents)"
        ],
        "artifacts": [],
    }
    return curl_post('/db/execute', data)


def apply(component):
    data = {'component': {component['dict']['identifier']: component}}
    return curl_post('/db/apply', data)


def delete():
    data = {
        "documents": [],
        "query": [
            "documents.delete_many({})"
        ],
        "artifacts": [],
    }
    return curl_post('/db/execute', data)


def remove(type_id, identifier):
    return curl_post('/db/remove?type_id={type_id}&identifier={identifier}', {})


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


def teardown():
    delete()
    remove('datatype', 'image')


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'setup':
        setup()
    elif sys.argv[1] == 'teardown':
        teardown()
    else:
        raise NotImplementedError