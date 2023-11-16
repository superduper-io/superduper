import base64
import json

import requests

from superduperdb import CFG, logging
from superduperdb.ext.utils import superduperencode


def request_server(
    service: str = 'vector_search', data=None, endpoint='add', args={}, type='post'
):
    url = getattr(CFG.server, service) + '/' + service + '/' + endpoint
    logging.debug(f'Trying to connect {service} at {url} method: {type}')
    if type == 'post':
        data = superduperencode(data)
        if isinstance(data, dict):
            if '_content' in data:
                data['_content']['bytes'] = base64.b64encode(
                    data['_content']['bytes']
                ).decode()
        response = requests.post(url, json=data, params=args)
        result = json.loads(response.content)
    else:
        response = requests.get(url, params=args)
        result = None
    if response.status_code != 200:
        error = json.loads(response.content)
        msg = f'Server error at {service} with {response.status_code} :: {error}'
        logging.error(msg)
        raise Exception(msg)
    return result
