"""
The component module provides the base class for all components in SuperDuperDB.
"""

from __future__ import annotations

import dataclasses as dc
import json
import os
import pprint
import re
import tempfile
import typing as t
from collections import defaultdict, namedtuple
from functools import wraps

from superduperdb import logging
from superduperdb.base.leaf import Leaf
from superduperdb.base.serializable import Serializable, _find_variables_with_path
from superduperdb.jobs.job import ComponentJob, Job
from superduperdb.misc.archives import from_tarball, to_tarball

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

    type_id: t.ClassVar[str]
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
        from superduperdb.components.datatype import _BaseEncodable

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

    @staticmethod
    def _import_from_references(references, db: t.Optional[Datalayer] = None, identifier: t.Optional[str] = None):
        if db is not None:
            all_references = _find_references(references['_components'])
            for k in all_references:
                # E.g. `$components/<type_id>/<identifier>/<version>`
                split = k.split('/')
                if split[0][1:] != '_components':
                    continue
                if len(split) == 4:
                    type_id, identifier, version = split[1:]
                    version = int(version)
                elif len(split) == 3:
                    type_id, identifier = split[1:]
                    version = None
                else:
                    raise Exception(f'Unexpected reference {k}')
                try:
                    references['_components'][k] = db.load(type_id, identifier, version)
                    logging.info(f'Loaded {k} from db')
                except Exception as e:
                    logging.warn(f'Failed to load {k} from db: {e}')

        while True:
            # extract and replace the references inside the component defitions
            for c in references['_components']:
                references['_components'][c] = _replace_references(
                    references['_components'][c], references
                )

            # Find the component definitions which don't refer to anything else
            # (leaf nodes)
            identifiers = [
                c
                for c in references['_components']
                if not _find_references(references['_components'][c])
                and isinstance(references['_components'][c], dict)
            ]

            if not identifiers:
                raise Exception(
                    f'Decompile failed due to unspecified references:'
                )

            for c in identifiers:
                r = references['_components'][c]
                references['_components'][c] = Serializable.decode(r)

            previous_identifiers = identifiers[:]

            # Stop when all components have been converted
            if not [
                c
                for c in references['_components']
                if isinstance(references['_components'][c], dict)
            ]:
                break

        if len(previous_identifiers) == 1:
            return references['_components'][previous_identifiers[0]]

        from superduperdb.components.stack import Stack

        return Stack(
            identifier or 'stack',
            components=[references['_components'][c] for c in previous_identifiers],
        )

    @staticmethod
    def _fix_format(definition, db):
        from superduperdb import Document

        def get_class(c):
            import importlib
            mod = importlib.import_module(c['module'])
            return getattr(mod, c['cls'])

        new_definition = {'_components': {}, '_artifacts': {}, '_lazy_artifacts': {}}
        for c in definition['_components']:
            k = f'{c["type_id"]}/{c["dict"]["identifier"]}'
            c['dict'] = get_class(c).handle_integration(c['dict'])
            new_definition['_components'][k] = c

        for c in definition.get('_artifacts', {}):
            new_definition['_artifacts'][c['_content']['file_id']] = c

        for c in definition.get('_lazy_artifacts', {}):
            new_definition['_lazy_artifacts'][c['_content']['file_id']] = c

        return new_definition

    @staticmethod
    def import_from_references(definition: t.Dict, db: Datalayer, identifier: t.Optional[str] = None):
        from superduperdb import Document
        from superduperdb.components.datatype import LazyArtifact

        definition = Component._fix_format(definition, db)

        if definition.get('_lazy_artifacts'):
            for a in definition['_lazy_artifacts']:
                definition['_lazy_artifacts'][a] = LazyArtifact(
                    **definition['_lazy_artifacts'][a]['_content']
                )
            definition['_lazy_artifacts'] = dict(
                Document.decode(definition['_lazy_artifacts'], db=db)
            )
        if definition.get('_artifacts'):
            definition['_artifacts'] = dict(
                Document.decode(definition['_artifacts'], db=db)
            )
        return Component._import_from_references(definition, db=db, identifier=identifier)

    def export_to_references(self):
        from superduperdb.base.document import _encode_with_references

        def process_subpart(r, references):
            if isinstance(r, dict):
                for k in list(r.keys())[:]:
                    r[k] = process_subpart(r[k], references)
            if isinstance(r, Component):
                r = dict(r.dict())
                _encode_with_references(r, references)
                return process_subpart(r, references)
            if isinstance(r, list):
                for i, x in enumerate(r):
                    r[i] = process_subpart(x, references)
            return r

        references = defaultdict(dict)
        
        try:
            references['_components'] = {self.unique_id: self}
        except Exception:
            references['_components'] = {f'{self.type_id}/{self.identifier}': self}
        while True:
            keys = list(references['_components'].keys())[:]
            for k in keys:
                if isinstance(references['_components'][k], Component):
                    references['_components'][k] = process_subpart(
                        references['_components'][k], references
                    )
            if all(isinstance(v, dict) for v in references['_components'].values()):
                break

        references['_components'] = [v for v in references['_components'].values()]
        references['_artifacts'] = [v for v in references['_artifacts'].values()]
        references['_lazy_artifacts'] = [v for v in references['_lazy_artifacts'].values()]
        return references


def _find_references(r):
    if isinstance(r, str):
        if r.startswith('$'):
            return [r]
    refs = []
    if isinstance(r, dict):
        for k, v in r.items():
            refs.extend(_find_references(v))
    if isinstance(r, list):
        for x in r:
            refs.extend(_find_references(x))
    return refs


def _replace_references(r, lookup, raises=False):
    if isinstance(r, str) and r.startswith('$'):
        leaf_type, key = re.match('\$(.*?)\/(.*)$', r).groups()
        try:
            if isinstance(lookup[leaf_type][key], dict):
                return r
            return lookup[leaf_type][key]
        except KeyError as e:
            if raises:
                raise e
            else:
                return r
    if isinstance(r, dict):
        for k, v in r.items():
            r[k] = _replace_references(v, lookup)
    if isinstance(r, list):
        for i, x in enumerate(r):
            r[i] = _replace_references(x, lookup)
    return r


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