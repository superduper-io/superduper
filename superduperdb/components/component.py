"""
The component module provides the base class for all components in SuperDuperDB.
"""

from __future__ import annotations

import dataclasses as dc
import typing as t
from collections import namedtuple
from functools import wraps

from superduperdb import logging
from superduperdb.base.leaf import Leaf
from superduperdb.base.serializable import Serializable, _find_variables_with_path
from superduperdb.jobs.job import ComponentJob, Job

if t.TYPE_CHECKING:
    from superduperdb import Document
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.dataset import Dataset
    from superduperdb.components.datatype import DataType


def getdeepattr(obj, attr):
    for a in attr.split('.'):
        obj = getattr(obj, a)
    return obj


ComponentTuple = namedtuple('ComponentTuple', ['type_id', 'identifier', 'version'])


@dc.dataclass
class Component(Serializable, Leaf):
    """
    :param identifier: A unique identifier for the component"""

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
        return kwargs

    @property
    def id(self):
        if self.version is None:
            return f'_component/{self.type_id}/{self.identifier}'
        else:
            return f'_component/{self.type_id}/{self.identifier}/{self.version}'

    @property
    def id_tuple(self):
        return ComponentTuple(self.type_id, self.identifier, self.version)

    def __post_init__(self, artifacts):
        self.artifacts = artifacts
        self.version: t.Optional[int] = None
        self._db = None
        if not self.identifier:
            raise ValueError('identifier cannot be empty or None')

    @classmethod
    def get_ui_schema(cls):
        out = {}
        ancestors = cls.mro()[::-1]
        for a in ancestors:
            if hasattr(a, 'ui_schema'):
                out.update({x['name']: x for x in a.ui_schema})
        return list(out.values())

    def set_variables(self, db, **kwargs):
        """
        Set free variables of self.

        :param db:
        """

        r = self.dict()
        variables = _find_variables_with_path(r['dict'])
        for r in variables:
            v = r['variable']
            value = v.set(db=db, **kwargs)
            self.setattr_with_path(r['path'], value)

    @property
    def dependencies(self):
        return ()

    def init(self):
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
    def db(self):
        return self._db

    @db.setter
    def db(self, value: Datalayer):
        self._db = value

    def pre_create(self, db: Datalayer) -> None:
        """Called the first time this component is created

        :param db: the db that creates the component
        """
        assert db

    def post_create(self, db: Datalayer) -> None:
        """Called after the first time this component is created.
        Generally used if ``self.version`` is important in this logic.

        :param db: the db that creates the component
        """
        assert db

    def on_load(self, db: Datalayer) -> None:
        """Called when this component is loaded from the data store

        :param db: the db that loaded the component
        """
        assert db

    def dict(self) -> 'Document':
        from superduperdb import Document
        from superduperdb.components.datatype import Artifact, File

        r = Document(super().dict())
        s = self.artifact_schema
        for k in s.fields:
            attr = getattr(self, k)
            if isinstance(attr, (Artifact, File)):
                r[f'dict.{k}'] = attr
            else:
                if s.fields[k].encodable == 'file':
                    r[f'dict.{k}'] = s.fields[k](uri=attr)  # artifact or file
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
        r = super().encode(leaf_types_to_keep=leaf_types_to_keep)
        del r['_content']['dict']
        r['_content']['leaf_type'] = 'component'
        return r

    @classmethod
    def decode(cls, r, db: t.Optional[t.Any] = None, reference: bool = False):
        assert db is not None
        r = r['_content']
        assert r['version'] is not None
        return db.load(r['type_id'], r['identifier'], r['version'], allow_hidden=True)

    @property
    def unique_id(self) -> str:
        if getattr(self, 'version', None) is None:
            raise Exception('Version not yet set for component uniqueness')
        return f'{self.type_id}/{self.identifier}/{self.version}'

    def create_validation_job(
        self,
        validation_set: t.Union[str, Dataset],
        metrics: t.Sequence[str],
    ) -> ComponentJob:
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
        """Run the job for this listener

        :param database: The db to process
        :param dependencies: A sequence of dependencies,
        :param verbose: If true, print more information
        """
        return []

    @classmethod
    def make_unique_id(cls, type_id: str, identifier: str, version: int) -> str:
        return f'{type_id}/{identifier}/{version}'

    def __setattr__(self, k, v):
        if k in dc.fields(self):
            self.changed.add(k)
        return super().__setattr__(k, v)


def ensure_initialized(func):
    """Decorator to ensure that the model is initialized before calling the function"""

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
