from functools import wraps
from superduperdb.datalayer.base import database
import dataclasses as dc
import requests
import superduperdb as s


@dc.dataclass
class Client:
    cfg: s.config.Server.WebServer = dc.field(default_factory=s.CFG.server.deepcopy)

    def check_health(self) -> None:
        response = requests.get('/health')
        assert response.status_code == 200
        assert response.text == 'ok'

    def download(self, uri_document: str) -> bytes:
        response = requests.get(f'/download/{uri_document}')
        assert response.status_code == 200
        return response.content


def _add_endpoint(name):
    # TODO: Formalize this into something reliable
    Result = getattr(database, f'{name.capitalize()}Result', database.UpdateResult)

    @wraps(getattr(database.BaseDatabase, name))
    def method(self, command):
        kwargs = requests.post(f'/{name}', json=command.dict()).json()
        return Result(**kwargs)

    setattr(Client, name, method)


[_add_endpoint(name) for name in database.ENDPOINTS]
