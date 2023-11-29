import base64
import json
from functools import lru_cache

import requests

from superduperdb import CFG, logging
from superduperdb.base import exceptions
from superduperdb.ext.utils import superduperencode


@lru_cache(maxsize=None)
def _handshake(service: str):
    endpoint = 'handshake/config'
    cfg = json.dumps(CFG.comparables)
    try:
        _request_server(service, args={'cfg': cfg}, endpoint=endpoint)
    except Exception:
        raise Exception("Incompatible configuration")


def _request_server(
    service: str = 'vector_search', data=None, endpoint='add', args={}, type='post'
):
    url = getattr(CFG.cluster, service) + '/' + endpoint
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
        raise exceptions.ServiceRequestException(msg)
    return result


def request_server(
    service: str = 'vector_search', data=None, endpoint='add', args={}, type='post'
):
    _handshake(service)
    return _request_server(
        service=service, data=data, endpoint=endpoint, args=args, type=type
    )
