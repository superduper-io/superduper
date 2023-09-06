import dataclasses as dc
import inspect
import logging
import typing as t
import uuid

import click
import requests

from superduperdb.container.artifact_tree import (
    get_artifacts,
    load_artifacts,
    replace_artifacts_with_dict,
)
from superduperdb.container.component import Component
from superduperdb.container.document import Document, dump_bsons, load_bson, load_bsons
from superduperdb.container.serializable import Serializable
from superduperdb.db.base.db import ExecuteQuery
from superduperdb.db.base.query import Delete, Insert, Like, Select, SelectOne, Update
from superduperdb.misc.serialization import serializers


class ClientArtifactStore:
    def __init__(self, get, put, request_id):
        self.request_id = request_id
        self.get = get
        self.put = put

    def load_artifact(self, file_id, serializer, info=None):
        bytes = self.get(self.request_id, file_id)
        if info in inspect.signature(serializers[serializer]).parameters:
            return serializers[serializer].decode(bytes, info=info)
        else:
            return serializers[serializer].decode(bytes)

    def create_artifact(self, bytes):
        d = self.put(self.request_id, bytes).json()
        return d['file_id'], d['sha1']


class Client:
    def __init__(self, uri, requests=requests):
        """
        :param uri: uri of the server
        """
        self.uri = uri
        self.encoders = LoadDict(self, 'encoder')
        self.requests = requests

    def execute(self, query: ExecuteQuery):
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
        """
        Send a request to the server to execute a select query and then
        retrieve the data from the server.

        The request is split into three parts:
        1. Send the serialized query to the server for execution
        2. Retrieve the serialized documents from the server
        3. Deserialize the documents

        Each serialized query contains the following information:
        - request_id: id of the request, randomly generated
        - query: serialized query type

        The request for the serialized documents contains the following information:
        - request_id: id of the request, randomly generated
        - file_id: id of the documents, returned by the initial server response

        :param query: query type to be sent to the server

        :return: deserialized documents
        """
        request_id = str(uuid.uuid4())
        response = self._make_get_request(
            'select',
            json={
                'query': query.serialize(),
                'request_id': request_id,
            },
        ).json()
        result = self._get(request_id=request_id, file_id=response['file_id'])
        documents = load_bsons(result, encoders=self.encoders)
        return documents

    def insert(self, query: Insert):
        """
        Dump a sequence of documents into BSON and send to the server.

        The request is split into two parts:
        1. Send serialized documents to the server in a single request
        2. Send serialized metadata to the server, including the query type

        Each serialized sequence of documents request contains the following
        information:
        - request_id: id of the request, randomly generated
        - file_id: extra randomly generated id (TODO: redundant?)
        - data: serialized documents

        The metadata request contains the following information:
        - request_id: id of the request, randomly generated
        - query: serialized query type
        - documents: ID for the documents on the server

        :param query: query type and attached documents to be sent to the server
        """
        documents = dump_bsons(query.documents)
        query.documents = []
        serialized = query.serialize()
        file_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        self._put(request_id=request_id, file_id=file_id, data=documents)
        return self._make_post_or_put_request(
            'insert',
            method='POST',
            json={
                'query': serialized,
                'documents': file_id,
                'request_id': request_id,
            },
        )

    def delete(self, query: Delete):
        """
        Send serialized delete query to the server

        :param query: delete query to be sent to the server
        """
        self._make_post_or_put_request(
            'delete', method='POST', json={'query': query.serialize()}
        )
        return 'ok'

    def _get(self, request_id, file_id):
        """
        Retrieve serialized artifact from the server

        :param request_id: id of the request that created the artifact on the server
        :param file_id: id of the artifact

        :return: serialized artifact
        """
        response = self._make_get_request(
            f'artifacts/get/{request_id}/{file_id}',
        )
        return response.content

    def _put(self, request_id, file_id, data):
        """
        Send serialized artifact to the server

        :param request_id: id of the request, randomly generated
        :param file_id: id of the artifact, randomly generated
        :param data: serialized artifact
        """
        response = self._make_post_or_put_request(
            f'artifacts/put/{request_id}/{file_id}',
            method='PUT',
            data=data,
        )
        return response.text

    def add(self, component: Component):
        """
        Serialize component to metadata and artifacts and send to the server.

        The request is split into two parts:
        1. Send serialized artifacts to the server one-by-one
        2. Send serialized metadata to the server

        Each artifact request contains the following information:
        - request_id: id of the request, randomly generated
        - file_id: id of the file, randomly generated
        - data: serialized artifact

        The metadata request contains the following information:
        - request_id: id of the request, randomly generated
        - component: serialized metadata
        - serializers: mapping from file_id to serializer name eg 'dill'

        :param component: component to be serialized and sent to the server
        """
        request_id = str(uuid.uuid4())
        d = component.serialize()
        artifacts = set(get_artifacts(d))
        lookup = {a: str(uuid.uuid4()) for a in artifacts}
        serializers = {lookup[a]: a.serializer for a in artifacts}
        d = replace_artifacts_with_dict(d, lookup)
        for a in artifacts:
            self._put(
                request_id=request_id,
                file_id=lookup[a],
                data=a.serialize(),  # Do we need to serialize again?
            )
        self._make_post_or_put_request(
            'add',
            method='POST',
            json={
                'component': d,
                'serializers': serializers,
                'request_id': request_id,
            },
        )

    def show(
        self,
        type_id: str,
        identifier: t.Optional[str] = None,
        version: t.Optional[int] = None,
    ):
        """
        Send show request to the server

        :param type_id: type_id of component to show ['encoder', 'model', 'listener',
                       'learning_task', 'training_configuration', 'metric',
                       'vector_index', 'job']
        :param identifier: identifying string to component
        :param version: (optional) numerical version - specify for full metadata

        :return: metadata of componentd already added to the database
        """
        return self._make_get_request(
            'show',
            json={
                'type_id': type_id,
                'identifier': identifier,
                'version': version,
            },
        ).json()

    def remove(
        self,
        type_id: str,
        identifier: str,
        version: t.Optional[int] = None,
        force: bool = False,
    ):
        """
        Send forced remove request to the server

        :param type_id: type_id of component to show ['encoder', 'model', 'listener',
                       'learning_task', 'training_configuration', 'metric',
                       'vector_index', 'job']
        :param identifier: identifying string to component
        :param version: (optional) numerical version - specify for full metadata
        """
        version_str = '' if version is None else f'/{version}'

        if not force and not click.confirm(
            f'You are about to delete {type_id}/{identifier}{version_str}'
            ', are you sure?',
            default=False,
        ):
            logging.info('aborting...')
            return

        self._make_post_or_put_request(
            'remove',
            method='POST',
            json={
                'type_id': type_id,
                'identifier': identifier,
                'version': version,
            },
        )

    def load(self, type_id: str, identifier: str, version: t.Optional[int] = None):
        """
        Load component from the database via the server.

        This endpoint sends a request to the server to load the component from the
        database. It then receives the component metadata as a response. Then it
        retrieves the component from the server, derserialises the component and
        returns it. It generates a random ID value for communicating with the server.

        :param type_id: type_id of component to load ['encoder', 'model', 'listener',
                       'learning_task', 'training_configuration', 'metric',
                       'vector_index', 'job']
        :param identifier: identifying string to component
        :param version: (optional) numerical version - specify for full metadata
        """
        request_id = str(uuid.uuid4())
        d = self._make_get_request(
            'load',
            json={
                'type_id': type_id,
                'identifier': identifier,
                'version': version,
                'request_id': request_id,
            },
        ).json()
        d = load_artifacts(
            d, cache={}, getter=lambda x: self._get(request_id=request_id, file_id=x)
        )
        return Serializable.deserialize(d)

    def select_one(self, query: SelectOne) -> Document:
        """
        Send a select query to the server and return a the result.

        This endpoint sends a request to the server to execute a single select
        statement in the database. It receives an ID for the result from the
        server, and then uses this ID to retrieve the result from the server.
        Finally it deserializes the result and returns it. It generates a random
        ID value for communicating with the server.

        :param query: serialized query
        """
        request_id = str(uuid.uuid4())
        response = self._make_get_request(
            'select_one',
            json={
                'query': query.serialize(),
                'request_id': request_id,
            },
        ).json()
        result = self._get(request_id=request_id, file_id=response['file_id'])
        return load_bson(result, encoders=self.encoders)

    def like(self, query: Like):
        # TODO: Implement server functionality for this endpoint.
        # TODO: Like.like does not exist, and there is no candidate to replace it
        like = query.like.dump_bson()  # type: ignore[attr-defined]
        query.like = None  # type: ignore[attr-defined]
        serialized = query.serialize()
        file_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        self._put(request_id=request_id, file_id=file_id, data=like)
        out = self._make_get_request(
            'like',
            json={
                'query': serialized,
                'like': file_id,
                'request_id': request_id,
            },
        ).json
        results = self._get(request_id=request_id, file_id=out['file_id'])
        return load_bsons(results, encoders=self.encoders)

    def update(self, query: Update):
        """
        Dump a sequence of documents into BSON and send to the server.

        The request is split into two parts:
        1. Send serialized documents to the server in a single request
        2. Send serialized metadata to the server, including the query type

        Each serialized sequence of documents request contains the following
        information:
        - request_id: id of the request, randomly generated
        - file_id: extra randomly generated id (TODO: redundant?)
        - data: serialized documents

        The metadata request contains the following information:
        - request_id: id of the request, randomly generated
        - query: serialized query type
        - documents: ID for the documents on the server

        :param query: query type and attached documents to be sent to the server
        """
        # TODO: Not sure that this even works...
        # TODO: Update.update does not exist, and there is no candidate to replace it
        update = query.update.dump_bson()  # type: ignore[attr-defined]

        file_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        self._put(request_id=request_id, file_id=file_id, data=update)
        query.update = None  # type: ignore[attr-defined]
        query.serialize()
        return self._make_post_or_put_request(
            'update',
            method='POST',
            json={
                'query': query.serialize(),
                'update': file_id,
                'request_id': request_id,
            },
        )

    def _make_get_request(
        self,
        route: str,
        params: t.Optional[t.Dict] = None,
        json: t.Optional[t.Dict] = None,
    ):
        response = self.requests.get(f'{self.uri}/{route}', params=params, json=json)
        print('YYYYY', response)
        if response.status_code != 200:
            raise ServerSideException(
                f'HTTP status {response.status_code} while making request to'
                f' /{route} with params {params} and json {json}:\n'
                f'{response.text}'
            )
        return response

    def _make_post_or_put_request(
        self,
        route: str,
        method: str,
        json: t.Optional[t.Dict] = None,
        data: t.Optional[t.Dict] = None,
    ):
        if method not in ['PUT', 'POST']:
            raise ServerSideException('Only PUT or POST methods are supported')
        fn = getattr(self.requests, method.lower())
        response = fn(
            f'{self.uri}/{route}',
            data=data,
            json=json,
        )
        if response.status_code != 200:
            raise ServerSideException(
                f'Non 200 status while making request to'
                f' /{route} with json {json}:\n'
                f'{response.text}'
            )
        return response


class ServerSideException(ValueError):
    pass


@dc.dataclass
class LoadDict(dict):
    client: Client
    field: str

    def __missing__(self, key: str):
        value = self[key] = self.client.load(self.field, key)
        return value
