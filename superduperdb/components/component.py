"""The component module provides the base class for all components in SuperDuperDB."""

from __future__ import annotations

import dataclasses as dc
import json
import os
import typing as t
from collections import namedtuple
from functools import wraps

import yaml

from superduperdb import logging
from superduperdb.base.leaf import Leaf
from superduperdb.jobs.job import ComponentJob, Job
from superduperdb.misc.annotations import merge_docstrings

if t.TYPE_CHECKING:
    from superduperdb import Document
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.dataset import Dataset
    from superduperdb.components.datatype import DataType


def _build_info_from_path(path: str):
    try:
        config = os.path.join(path, 'component.json')
        with open(config) as f:
            config_object = json.load(f)
    except FileNotFoundError:
        try:
            config = os.path.join(path, 'component.yaml')
            with open(config) as f:
                config_object = yaml.safe_load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError('No component.json or component.yaml found') from e

    config_object['_blobs'] = {}
    if os.path.exists(os.path.join(path, 'blobs')):
        blobs = {}
        for file_id in os.listdir(os.path.join(path, 'blobs')):
            with open(os.path.join(path, 'blobs', file_id), 'rb') as f:
                blobs[file_id] = f.read()
        config_object['_blobs'] = blobs
    return config_object


def import_(r=None, path=None, db=None):
    """Helper function for importing component JSONs, YAMLs, etc.

    :param r: Object to be imported.
    :param path: Components directory.
    :param db: Datalayer instance.
    """
    from superduperdb.base.document import _build_leaves

    if r is None:
        try:
            with open(f'{path}/component.json') as f:
                r = json.load(f)
        except FileNotFoundError:
            with open(f'{path}/component.yaml') as f:
                r = yaml.safe_load(f)
        for id_ in os.listdir(path):
            if id_ == 'component.json' or id_ == 'component.yaml':
                continue
            with open(f'{path}/{id_}', 'rb') as f:
                bytes[id_] = f.read()
    r['_leaves'] = _build_leaves(r['_leaves'], db=db)[0]
    return r['_leaves'][r['_base']]


def getdeepattr(obj, attr):
    """Get nested attribute with dot notation.

    :param obj: Object.
    :param attr: Attribute.
    """
    for a in attr.split('.'):
        obj = getattr(obj, a)
    return obj


ComponentTuple = namedtuple('ComponentTuple', ['type_id', 'identifier', 'version'])
ComponentTuple.__doc__ = 'noqa'


@merge_docstrings
@dc.dataclass(kw_only=True)
class Component(Leaf):
    """Base class for all components in SuperDuperDB.

    Class to represent SuperDuperDB serializable entities
    that can be saved into a database.

    :param artifacts: A dictionary of artifacts paths and `DataType` objects
    """

    type_id: t.ClassVar[str] = 'component'
    leaf_type: t.ClassVar[str] = 'component'
    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, 'DataType']]] = ()
    set_post_init: t.ClassVar[t.Sequence] = ('version',)
    changed: t.ClassVar[set] = set([])

    artifacts: dc.InitVar[t.Optional[t.Dict]] = None

    @property
    def _id(self):
        return f':component:{self.type_id}/{self.identifier}/{self.uuid}'.replace(
            '.', '-'
        )

    @property
    def leaves(self):
        return {}

    def __post_init__(self, db, artifacts):
        super().__post_init__(db)

        self.artifacts = artifacts
        self.version: t.Optional[int] = None
        if not self.identifier:
            raise ValueError('identifier cannot be empty or None')

    def cleanup(self, db: Datalayer):
        """Method to clean the component."""
        pass

    @property
    def id(self):
        """Returns the component identifier."""
        return f'component/{self.type_id}/{self.identifier}/{self.uuid}'

    # TODO remove
    @property
    def id_tuple(self):
        """Returns an object as `ComponentTuple`."""
        return ComponentTuple(self.type_id, self.identifier, self.version)

    @property
    def dependencies(self):
        """Get dependencies on the component."""
        return ()

    def init(self, db=None):
        """Method to help initiate component field dependencies."""
        self.db = self.db or db
        self.unpack(db=db)

    def unpack(self, db=None):
        """Method to unpack the component.

        This method is used to initialize all the fields of the component and leaf
        """

        def _init(item):
            nonlocal db
            if isinstance(item, Component):
                item.init(db=db)
                return item

            if isinstance(item, dict):
                return {k: _init(i) for k, i in item.items()}

            if isinstance(item, list):
                return [_init(i) for i in item]

            if isinstance(item, Leaf):
                item.init(db=db)
                return item.unpack()

            return item

        for f in dc.fields(self):
            item = getattr(self, f.name)
            unpacked_item = _init(item)
            setattr(self, f.name, unpacked_item)

        return self

    @property
    def artifact_schema(self):
        """Returns `Schema` representation for the serializers in the component."""
        from superduperdb import Schema
        from superduperdb.components.datatype import dill_serializer

        schema = {}
        lookup = dict(self._artifacts)
        if self.artifacts is not None:
            lookup.update(self.artifacts)
        for f in dc.fields(self):
            a = getattr(self, f.name)
            if a is None:
                continue
            if f.name in lookup:
                schema[f.name] = lookup[f.name]
                continue
            if isinstance(getattr(self, f.name), Component):
                continue
            if callable(getattr(self, f.name)) and not isinstance(
                getattr(self, f.name), Leaf
            ):
                schema[f.name] = dill_serializer
        return Schema(identifier=f'serializer/{self.identifier}', fields=schema)

    @property
    def db(self) -> Datalayer:
        """Datalayer instance."""
        return self._db

    @db.setter
    def db(self, value: Datalayer):
        """Datalayer instance property setter.

        :param value: Datalayer instance to set.

        """
        self._db = value

    def pre_create(self, db: Datalayer) -> None:
        """Called the first time this component is created.

        :param db: the db that creates the component.
        """
        assert db

    def post_create(self, db: Datalayer) -> None:
        """Called after the first time this component is created.

        Generally used if ``self.version`` is important in this logic.

        :param db: the db that creates the component.
        """
        assert db

    def on_load(self, db: Datalayer) -> None:
        """Called when this component is loaded from the data store.

        :param db: the db that loaded the component.
        """
        assert db

    def _deep_flat_encode(self, cache, blobs, files, leaves_to_keep=(), schema=None):
        if isinstance(self, leaves_to_keep):
            cache[self._id] = self
            return f'?{self._id}'
        from superduperdb.base.document import _deep_flat_encode

        r = dict(self.dict())
        r = _deep_flat_encode(
            r,
            cache,
            blobs,
            files,
            leaves_to_keep=leaves_to_keep,
            schema=schema,
        )
        cache[self._id] = r
        return f'?{self._id}'

    @staticmethod
    def read(path: str):
        """
        Read a `Component` instance from a directory created with `.export`.

        :param path: Path to the directory containing the component.

        Expected directory structure:
        ```
        |_component.json/yaml
        |_blobs/*
        |_files/*
        ```
        """
        config_object = _build_info_from_path(path=path)

        from superduperdb import Document

        return Document.decode(config_object).unpack()

    def export(self, path: str, format: str = 'json'):
        """
        Save `self` to a directory using super-duper protocol.

        :param path: Path to the directory to save the component.

        Created directory structure:
        ```
        |_component.json/yaml
        |_blobs/*
        |_files/*
        ```
        """
        r = self.encode()
        os.makedirs(path, exist_ok=True)
        if r.blobs:
            os.makedirs(os.path.join(path, 'blobs'), exist_ok=True)
            for file_id, bytestr_ in r.blobs.items():
                filepath = os.path.join(path, 'blobs', file_id)
                with open(filepath, 'wb') as f:
                    f.write(bytestr_)

        r.pop_blobs()
        if format == 'json':
            with open(os.path.join(path, 'component.json'), 'w') as f:
                json.dump(r, f, indent=2)
        elif format == 'yaml':
            with open(os.path.join(path, 'component.yaml'), 'w') as f:
                json.dump(r, f, indent=2)

        from superduperdb import REQUIRES

        with open(os.path.join(path, 'requirements.txt'), 'w') as f:
            f.write('\n'.join(REQUIRES))

    def dict(self) -> 'Document':
        """A dictionary representation of the component."""
        from superduperdb import Document
        from superduperdb.components.datatype import Artifact, File

        r = super().dict()
        s = self.artifact_schema
        for k in s.fields:
            attr = getattr(self, k)
            if isinstance(attr, (Artifact, File)):
                r[k] = attr
            else:
                r[k] = s.fields[k](x=attr)  # artifact or file

        r['type_id'] = self.type_id
        r['version'] = self.version
        r['identifier'] = self.identifier
        r['hidden'] = False
        return Document(r)

    @classmethod
    def decode(cls, r, db: t.Optional[t.Any] = None, reference: bool = False):
        """Decodes a dictionary component into a `Component` instance.

        :param r: Object to be decoded.
        :param db: Datalayer instance.
        :param reference: If decode with reference.
        """
        assert db is not None
        r = r['_content']
        assert r['version'] is not None
        return db.load(r['type_id'], r['identifier'], r['version'], allow_hidden=True)

    def create_validation_job(
        self,
        validation_set: t.Union[str, Dataset],
        metrics: t.Sequence[str],
    ) -> ComponentJob:
        """Method to create a validation job with `validation_set` and `metrics`.

        :param validation_set: Kwargs for the `predict` method of `Model`.
        :param metrics: Kwargs for the `predict` method of `Model` to set
                        metrics for the validation job.
        """
        assert self.identifier is not None
        return ComponentJob(
            component_identifier=self.identifier,
            method_name='predict',
            type_id='model',
            kwargs={
                'validation_set': validation_set,
                'metrics': metrics,
            },
        )

    def schedule_jobs(
        self,
        db: Datalayer,
        dependencies: t.Sequence[Job] = (),
    ) -> t.Sequence[t.Any]:
        """Run the job for this listener.

        :param db: The db to process.
        :param dependencies: A sequence of dependencies.
        """
        return []

    def __setattr__(self, k, v):
        if k in dc.fields(self):
            self.changed.add(k)
        return super().__setattr__(k, v)

    def info(self):
        from superduperdb.misc.special_dicts import display_component
        display_component(self)


def ensure_initialized(func):
    """Decorator to ensure that the model is initialized before calling the function.

    :param func: Decorator function.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_is_initialized") or not self._is_initialized:
            model_message = f"{self.__class__.__name__} : {self.identifier}"
            logging.debug(f"Initializing {model_message}")
            self.init()
            self._is_initialized = True
            logging.debug(f"Initialized  {model_message} successfully")
        return func(self, *args, **kwargs)

    return wrapper
