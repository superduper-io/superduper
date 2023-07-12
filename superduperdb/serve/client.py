import dataclasses as dc
import io
import typing as t
import uuid

import click
import requests

from superduperdb.core.component import Component
from superduperdb.core.documents import Document, ArtifactDocument
from superduperdb.core.serializable import Serializable
from superduperdb.datalayer.base.database import ExecuteQuery
from superduperdb.datalayer.base.query import Delete, Insert, Select, Like, SelectOne, Update
from superduperdb.misc.serialization import serializers
from superduperdb.datalayer.base.database import BaseDatabase


class ClientArtifactStore:
    def __init__(self, get, put, request_id):
        self.request_id = request_id
        self.get = get
        self.put = put

    def load_artifact(self, file_id, serializer, info=None):
        bytes = self.get(self.request_id, file_id)
        return serializers[serializer].decode(bytes, info=info)

    def create_artifact(self, bytes):
        d = self.put(self.request_id, bytes).json()
        return d['file_id'], d['sha1']


class Client:
    def __init__(self, uri):
        self.uri = uri
        self.encoders = LoadDict(self, 'encoder')

    def _make_get_request(
        self,
        route: str,
        params: t.Optional[t.Dict] = None,
    ):
        return requests.get(
            f'{self.uri}/{route}',
            params=params,
        )

    def _make_post_request(
        self,
        route: str,
        json: t.Optional[t.Dict] = None,
        data: t.Optional[t.Dict] = None,
        files: t.Optional[t.Dict] = None,
    ):
        return requests.post(
            f'{self.uri}/{route}',
            data=data,
            json=json,
            files=files,
        )

    def execute(self, query: ExecuteQuery):
        __doc__ = BaseDatabase.execute.__doc__

        if isinstance(query, Delete):
            return self.delete(query)
        if isinstance(query, Insert):
            return self.insert(query)
        if isinstance(query, Select):
            return self.select(query)
        if isinstance(query, Like):
            return self.like(query)
        if isinstance(query, SelectOne):
            return self.select_one(query)
        if isinstance(query, Update):
            return self.update(query)
        raise TypeError(
            f'Wrong type of {query}; '
            f'Expected object of type {t.Union[Select, Delete, Update, Insert]}; '
            f'Got {type(query)};'
        )

    def select(self, query: Select):
        __doc__ = BaseDatabase.select.__doc__
        response = self._make_post_request(
            f'select',
            json={
                'query': query.to_dict(),
            }
        )
        documents = Document.loads_many(response.content, encoders=self.encoders)
        return documents

    def insert(self, query: Select):
        __doc__ = BaseDatabase.insert.__doc__
        documents = Document.dumps_many(query.documents)
        query.documents = None
        serialized = query.to_dict()
        file_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        self._put(request_id=request_id, file_id=file_id, data=documents)
        return self._make_post_request(
            f'insert',
            json={
                'query': serialized,
                'documents': file_id,
                'request_id': request_id,
            },
        )

    def delete(self, query: Delete):
        __doc__ = BaseDatabase.delete.__doc__
        self._make_post_request(
            f'delete',
            json={
                'query': query.to_dict(),
            }
        )
        return 'ok'

    def _get(self, request_id, file_id):
        response = self._make_get_request(
            f'artifacts/get/{request_id}/{file_id}',
        )
        return response.content

    def _put(self, request_id, file_id, data):
        response = self._make_post_request(
            f'artifacts/put/{request_id}/{file_id}',
            data=data,
        )
        return response.text

    def add(self, component: Component):
        __doc__ = BaseDatabase.add.__doc__
        ad = ArtifactDocument(component.to_dict())
        request_id = str(uuid.uuid4())
        ad.save_artifacts(
            artifact_store=ClientArtifactStore(
                request_id=request_id, put=self._put, get=self._get,
            ),
            cache={},
        )
        return 'ok'

    def show(
        self,
        variety: str,
        identifier: t.Optional[str] = None,
        version: t.Optional[int] = None,
    ):
        __doc__ = BaseDatabase.show.__doc__
        return self._make_get_request('show', params={
            'variety': variety,
            'identifier': identifier,
            'version': version,
        }).json()

    def remove(
        self,
        variety: str,
        identifier: str,
        version: t.Optional[int] = None,
        force: bool = False,
    ):
        __doc__ = BaseDatabase.remove.__doc__
        version_str = '' if version is None else f'/{version}'

        if not force and not click.confirm(
            f'You are about to delete {variety}/{identifier}{version_str}, are you sure?',
            default=False,
        ):
            print('aborting...')
            return

        self._make_post_request(
            'remove',
            json={
                'variety': variety,
                'identifier': identifier,
                'version': version,
            },
        )

    def load(self, variety: str, identifier: str, version: t.Optional[int] = None):
        __doc__ = BaseDatabase.load.__doc__

        request_id = str(uuid.uuid4())
        response = self._make_post_request(
            f'load',
            data={
                'variety': variety,
                'identifier': identifier,
                'version': version,
                'request_id': request_id,
            }
        )

        document = ArtifactDocument(response.json())
        document.load_artifacts(
            artifact_store=ClientArtifactStore(
                request_id=request_id, get=self._get, put=self._put,
            ),
            cache = {},
        )
        return Serializable.from_dict(document.content)

    def select_one(self, query: SelectOne) -> t.Dict:
        __doc__ = BaseDatabase.select_one.__doc__

        response = self._make_post_request(
            f'select_one',
            data={
                'query': query.to_dict(),
            }
        )
        documents = Document.loads(response.content, encoders=self.encoders)
        return documents

    def update(self, query: Update):
        __doc__ = BaseDatabase.update.__doc__
        update = Document.dumps(query.update)
        query.update = None
        file_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        self._put(request_id=request_id, file_id=file_id, data=update)
        serialized = query.to_dict()
        serialized.update = None
        return self._make_post_request(
            'update',
            data={
                'query': query.to_dict(),
            },
        )


@dc.dataclass
class LoadDict(dict):
    client: Client
    field: str

    def __missing__(self, key: str):
        value = self[key] = self.client.load(self.field, key)
        return value
