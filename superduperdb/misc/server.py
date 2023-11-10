import json

import requests

from superduperdb import CFG


def request_server(
    service: str = 'vector_search', data=None, endpoint='add', args={}, type='post'
):
    url = getattr(CFG.server, service) + '/' + service + '/' + endpoint
    if type == 'post':
        response = requests.post(url, json=data, params=args)
        result = json.loads(response.content)
    else:
        response = requests.get(url, params=args)
        result = None
    if response.status_code != 200:
        error = json.loads(response.content)
        msg = f'Server error at {service} with {response.status_code} :: {error}'
        raise Exception(msg)
    return result
