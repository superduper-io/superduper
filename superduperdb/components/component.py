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
from superduperdb.base.serializable import Serializable, _find_variables_with_path
from superduperdb.jobs.job import ComponentJob, Job

if t.TYPE_CHECKING:
    from superduperdb import Document
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.dataset import Dataset
    from superduperdb.components.datatype import DataType


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


@dc.dataclass
class Component(Serializable, Leaf):
    """Base class for all components in SuperDuperDB.

    Class to represent SuperDuperDB serializable entities
    that can be saved into a database.

    :param identifier: A unique identifier for the component.
    :param artifacts: List of artifacts which represent entities that are
                      not serializable by default.
    """

    type_id: t.ClassVar[str] = 'component'
    leaf_type: t.ClassVar[str] = 'component'
    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, 'DataType']]] = ()
    set_post_init: t.ClassVar[t.Sequence] = ('version',)
    ui_schema: t.ClassVar[t.List[t.Dict]] = [{'name': 'identifier', 'type': 'str'}]
    identifier: str
    artifacts: dc.InitVar[t.Optional[t.Dict]] = None
    changed: t.ClassVar[set] = set([])

    @classmethod
    def handle_integration(cls, kwargs):
        """Abstract method for handling integration.

        :param kwargs: Integration kwargs.
        """
        return kwargs

    @property
    def id(self):
        """Returns the component identifier."""
        return f'_component/{self.type_id}/{self.identifier}'

    @property
    def id_tuple(self):
        """Returns an object as `ComponentTuple`."""
        return ComponentTuple(self.type_id, self.identifier, self.version)

    def __post_init__(self, artifacts):
        self.artifacts = artifacts
        self.version: t.Optional[int] = None
        self._db = None
        if not self.identifier:
            raise ValueError('Identifier cannot be empty or None')

    @classmethod
    def get_ui_schema(cls):
        """Helper method to get the UI schema."""
        out = {}
        ancestors = cls.mro()[::-1]
        for a in ancestors:
            if hasattr(a, 'ui_schema'):
                out.update({x['name']: x for x in a.ui_schema})
        return list(out.values())

    def set_variables(self, db, **kwargs):
        """Set free variables of self.

        :param db: Datalayer instance.
        """
        r = self.dict()
        variables = _find_variables_with_path(r['dict'])
        for r in variables:
            v = r['variable']
            value = v.set(db=db, **kwargs)
            self.setattr_with_path(r['path'], value)

    @property
    def dependencies(self):
        """Get dependencies on the component."""
        return ()

    def init(self):
        """Method to help initiate component field dependencies."""

        def _init(item):
            if isinstance(item, Component):
                item.init()
                return item

            if isinstance(item, dict):
                return {k: _init(i) for k, i in item.items()}

            if isinstance(item, list):
                return [_init(i) for i in item]

            if isinstance(item, Leaf):
                item.init(db=self.db)
                return item.unpack(db=self.db)

            return item

        for f in dc.fields(self):
            item = getattr(self, f.name)
            unpacked_item = _init(item)
            setattr(self, f.name, unpacked_item)

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
            elif callable(getattr(self, f.name)) and not isinstance(
                getattr(self, f.name), Serializable
            ):
                schema[f.name] = dill_serializer
        return Schema(f'serializer/{self.identifier}', fields=schema)

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

    def _deep_flat_encode(self, cache):
        from superduperdb.base.document import _deep_flat_encode

        r = dict(self.dict())
        r['dict'] = _deep_flat_encode(r['dict'], cache)
        r['id'] = self.id
        cache[self.id] = r
        return self.id

    def deep_flat_encode(self):
        """Encode cache with deep flattened structure."""
        cache = {}
        id = self._deep_flat_encode(cache)
        return {'_leaves': list(cache.values()), '_base': id}

    def _to_dict_and_bytes(self):
        r = self.deep_flat_encode()
        bytes = {}
        for leaf in r['_leaves']:
            if 'bytes' in leaf['dict']:
                bytes[leaf['dict']['file_id']] = leaf['dict']['bytes']
                del leaf['dict']['bytes']
        return r, bytes

    def export(self, format=None):
        """Method to export the component in the provided format.

        If format is None, the method exports the component in a dictionary.

        :param format: `json` and `yaml`.
        """
        r, bytes = self._to_dict_and_bytes()

        if format is None:
            return r, bytes

        if self.version is None:
            path = f'_component.{self.type_id}.{self.identifier}'
        else:
            path = f'_component.{self.type_id}.{self.identifier}.{self.version}'

        os.makedirs(path, exist_ok=True)

        for file_id in bytes:
            with open(f'{path}/{file_id}', 'wb') as f:
                f.write(bytes[file_id])

        if format == 'json':
            with open(f'{path}/component.json', 'w') as f:
                f.write(json.dumps(r, indent=4))
            return r, bytes

        if format == 'yaml':
            with open(f'{path}/component.yaml', 'w') as f:
                f.write(yaml.safe_dump(r))
            return r, bytes

        raise NotImplementedError(f'Format {format} not supported')

    def dict(self) -> 'Document':
        """A dictionary representation of the component."""
        from superduperdb import Document
        from superduperdb.components.datatype import Artifact, File

        r = Document(super().dict())
        s = self.artifact_schema
        for k in s.fields:
            attr = getattr(self, k)
            if isinstance(attr, (Artifact, File)):
                r[f'dict.{k}'] = attr
            else:
                r[f'dict.{k}'] = s.fields[k](x=attr)  # artifact or file
        r['type_id'] = self.type_id
        r['identifier'] = self.identifier
        r['version'] = self.version
        r['dict.version'] = self.version
        r['hidden'] = False
        return r

    def encode(
        self,
        leaf_types_to_keep: t.Sequence = (),
    ):
        """Method to encode the component into a dictionary.

        :param leaf_types_to_keep: Leaf types to be excluded from encoding.
        """
        r = super().encode(leaf_types_to_keep=leaf_types_to_keep)
        del r['_content']['dict']
        r['_content']['leaf_type'] = 'component'
        r['_content']['id'] = self.id
        return r

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

    @property
    def unique_id(self) -> str:
        """Method to get a unique identifier for the component."""
        if getattr(self, 'version', None) is None:
            raise Exception('Version not yet set for component uniqueness')
        return f'{self.type_id}/{self.identifier}/{self.version}'

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

    @classmethod
    def make_unique_id(cls, type_id: str, identifier: str, version: int) -> str:
        """Class method to create a unique identifier.

        :param type_id: Component type id.
        :param identifier: Unique identifier.
        :param version: Component version.
        """
        return f'{type_id}/{identifier}/{version}'

    def __setattr__(self, k, v):
        if k in dc.fields(self):
            self.changed.add(k)
        return super().__setattr__(k, v)


def ensure_initialized(func):
    """Decorator to ensure that the model is initialized before calling the function.

    :param func: Decorator function.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_is_initialized") or not self._is_initialized:
            model_message = f"{self.__class__.__name__} : {self.identifier}"
            logging.info(f"Initializing {model_message}")
            self.init()
            self._is_initialized = True
            logging.info(f"Initialized  {model_message} successfully")
        return func(self, *args, **kwargs)

    return wrapper
