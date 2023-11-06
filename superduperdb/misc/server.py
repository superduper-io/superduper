import json

import requests

from superduperdb import CFG


def request_server(
    service: str = 'vector_search', data=None, endpoint='add', args={}, type='post'
):
    service_config = getattr(CFG.server, service)
    url = service_config.host + str(service_config.port) + '/' + endpoint
    args = '&'.join([f'{k}={v}' for k, v in args.items()])
    if args:
        url += '?' + args
    if type == 'post':
        response = requests.post(url, json=data)
        result = json.loads(response.content)
    else:
        response = requests.get(url)
        result = None
    if response.status_code != 200:
        error = json.loads(response.content)
        msg = f'Server error at {service} with {response.status_code} :: {error}'
        raise Exception(msg)
    return result
