from .test_registry import setup_registry
from fastapi.testclient import TestClient
from pydantic import dataclasses as dc
from pydantic import Field
from dataclasses import InitVar, asdict, field
from superduperdb.server.server import Server
import io
import json
import pytest
import superduperdb as s

skip_obsolete = pytest.mark.skip(reason='Obsolete tests')


def setup_test_client():
    server = Server()
    test = setup_registry(server.register)

    test.client = TestClient(server.app)
    test.object = test.Object()
    test.server = server

    server.add_endpoints(test.object)

    return test


def test_basics():
    with setup_test_client().client as client:
        response = client.get('/')
        assert response.status_code == 200
        assert 'SuperDuperServer' in response.text

        response = client.get('/health')
        assert response.status_code == 200
        assert response.text == 'ok'

        response = client.get('/stats')
        assert response.status_code == 200
        assert response.json() == {'perhaps': 'redoc makes this pointless'}


def test_methods():
    with setup_test_client().client as client:
        response = client.post('/first', json={'one': 'three'})
        assert response.status_code == 200
        assert response.json() == {'one': 'three', 'two': 'two'}


@skip_obsolete
def test_execute():
    with setup_test_client().client as client:
        response = client.post('/execute?method=first', json=[{'one': 'three'}])
        assert response.status_code == 200, json.dumps(response.json(), indent=2)
        assert response.json() == {'one': 'three', 'two': 'two'}


class One(s.JSONable):
    one = 'one'


class Two(One):
    two = 'two'


class Three(One):
    two = 'three'


class Object:
    def first(self, one: One) -> Two:
        return Two(**asdict(one))

    def second(self, one: One, three: Three) -> One:
        return one


@skip_obsolete
def test_auto_register():
    obj = Object()
    server = Server()
    server.auto_register(obj)
    server.add_endpoints(obj)
    with TestClient(server.app) as client:
        response = client.post('/execute?method=first', json=[{'one': 'three'}])
        assert response.status_code == 200, json.dumps(response.json(), indent=2)
        assert response.json() == {'one': 'three', 'two': 'two'}


def test_download():
    test = setup_test_client()

    blob = bytes(range(256))
    test.server.artifact_store['test-key'] = blob

    with TestClient(test.server.app) as client:
        response = client.get('/download/test-key')
        assert response.status_code == 200
        assert response.content == blob


def test_upload():
    test = setup_test_client()

    blob = bytes(range(256))

    with TestClient(test.server.app) as client:
        fp = io.BytesIO(blob)
        files = {'file': ('test-key', fp, 'multipart/form-data')}
        response = client.post('/upload/test-key', files=files)
        assert response.status_code == 200, json.dumps(response.json(), indent=2)
        assert response.json() == {'created': 'test-key'}

        fp = io.BytesIO(blob)
        files = {'file': ('test-key', fp, 'multipart/form-data')}
        response = client.post('/upload/test-key', files=files)
        assert response.status_code == 200, json.dumps(response.json(), indent=2)
        assert response.json() == {'replaced': 'test-key'}

        response = client.get('/download/test-key')
        assert response.status_code == 200
        assert response.content == blob


@dc.dataclass
class Un:
    un: str = 'un'

    # These two unfortunately get JSONized
    nine: str = Field(default='ERROR', exclude=True)
    ten: str = field(default='ERROR', repr=False, compare=False)
    eleven: InitVar[str] = 'this goes up to'

    def __post_init__(self, eleven: str):
        self.seven = self.un + '-sept'
        self.eleven = eleven


UN = {
    'un': 'un',
    'nine': 'ERROR',
    'ten': 'ERROR',
}


@dc.dataclass
class Deux(Un):
    deux: str = 'deux'


@dc.dataclass
class Trois(Un):
    deux: str = 'trois'


class Objet:
    def premier(self, un: Un) -> Deux:
        return Deux(**asdict(un))

    def second(self, un: Un, trois: Trois) -> Un:
        return un


@dc.dataclass
class Inclus:
    ein: Un


def test_dataclasses():
    assert asdict(Un(eleven='HAHA!')) == UN
    assert asdict(Inclus(Un())) == {'ein': UN}
    assert Inclus(**{'ein': UN}) == Inclus(Un())


def test_dataclasses2():
    server = Server()

    client = TestClient(server.app)
    o = Objet()
    server = server

    server.register(o.premier)
    server.register(o.second)
    server.add_endpoints(o)
    with client as client:
        response = client.post('/premier', json={'un': 'trois'})
        assert response.status_code == 200
        assert response.json() == dict(UN, deux='deux', un='trois')


class Lemon(s.JSONable):
    un: Un

    def __post_init__(self):
        self.seven = self.un + '-sept'
