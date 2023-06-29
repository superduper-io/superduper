from functools import wraps
from superduperdb.datalayer.base import database
import dataclasses as dc
import requests
import superduperdb as s


def _endpoint(name):
    Result = getattr(database, f'{name.capitalize()}Result')

    @wraps(getattr(database.BaseDatabase, name))
    def method(self, command):
        kwargs = requests.post(f'/{name}', json=command.dict()).json()
        return Result(**kwargs)

    return method


@dc.dataclass
class Client:
    cfg: s.config.Server.WebServer = dc.field(default_factory=s.CFG.server.deepcopy)

    delete = _endpoint('delete')
    execute = _endpoint('execute')
    insert = _endpoint('insert')
    select = _endpoint('select')
    update = _endpoint('update')

    def check_health(self) -> None:
        response = requests.get('/health')
        assert response.status_code == 200
        assert response.text == 'ok'

    def download(self, uri_document: str) -> bytes:
        response = requests.get(f'/download/{uri_document}')
        assert response.status_code == 200
        return response.content
