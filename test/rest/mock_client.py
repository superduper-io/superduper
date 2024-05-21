import json
import os
from urllib.parse import urlencode

from superduperdb import CFG

HOST = CFG.cluster.rest.uri
VERBOSE = os.environ.get('SUPERDUPERDB_VERBOSE', '1')


def make_params(params):
    return '?' + urlencode(params)


def curl_get(endpoint, params=None):
    if params is not None:
        params = make_params(params)
    else:
        params = ''
    request = f"curl '{HOST}{endpoint}{params}'"
    if VERBOSE == '1':
        print('CURL REQUEST:')
        print(request)
    result = os.popen(request).read()
    assert result, f'GET request to {request} returned empty response'
    result = json.loads(result)
    if 'msg' in result:
        raise Exception('Error: ' + result['msg'])
    return result


def curl_post(endpoint, data, params=None):
    if params is not None:
        params = make_params(params)
    else:
        params = ''
    data = json.dumps(data)
    request = f"""curl -X 'POST' \
        '{HOST}{endpoint}{params}' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -s \
        -d '{data}'"""
    if VERBOSE == '1':
        print('CURL REQUEST:')
        print(request)
    result = os.popen(request).read()
    assert result, f'POST request to {endpoint} with {data} returned empty response'
    result = json.loads(result)
    if 'msg' in result:
        raise Exception('Error: ' + result['msg'])
    return result


def curl_put(endpoint, file, media_type, params=None):
    if params is not None:
        params = make_params(params)
    else:
        params = ''
    request = f"""curl -X 'PUT' \
        '{HOST}{endpoint}{params}' \
        -H 'accept: application/json' \
        -H 'Content-Type: multipart/form-data' \
        -s \
        -F 'raw=@{file};type={media_type}'"""
    print('CURL REQUEST:')
    print(request)
    result = os.popen(request).read()
    assert (
        result
    ), f'PUT request to {endpoint} with {params} and {file} returned empty response'
    result = json.loads(result)
    return result


def insert(data):
    query = {'query': 'coll.insert_many(documents)', 'documents': data}
    return curl_post('/db/execute', data=query)


def apply(component):
    return curl_post('/db/apply', data=component)


def delete():
    return curl_post('/db/execute', data={'query': 'coll.delete_many({})'})


def remove(type_id, identifier):
    return curl_post(f'/db/remove?type_id={type_id}&identifier={identifier}', {})


def setup():
    data = [
        {"x": [1, 2, 3, 4, 5], "y": 'test'},
        {"x": [6, 7, 8, 9, 10], "y": 'test'},
    ]
    insert(data)


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
