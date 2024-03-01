"""
The component module provides the base class for all components in SuperDuperDB.
"""

from __future__ import annotations

import dataclasses as dc
import json
import os
import re
import tempfile
import typing as t
from collections import defaultdict
from functools import wraps

from superduperdb import logging
from superduperdb.base.leaf import Leaf
from superduperdb.base.serializable import Serializable
from superduperdb.jobs.job import ComponentJob, Job
from superduperdb.misc.archives import from_tarball, to_tarball

if t.TYPE_CHECKING:
    from superduperdb import Document
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.dataset import Dataset
    from superduperdb.components.datatype import DataType


@dc.dataclass
class Component(Serializable, Leaf):
    """
    :param identifier: A unique identifier for the component"""

    type_id: t.ClassVar[str]
    leaf_type: t.ClassVar[str] = 'component'
    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, 'DataType']]] = ()
    set_post_init: t.ClassVar[t.Sequence] = ('version',)
    identifier: str
    artifacts: dc.InitVar[t.Optional[t.Dict]] = None

    def __post_init__(self, artifacts):
        self.artifacts = artifacts
        self.version: t.Optional[int] = None
        self._db = None
        if not self.identifier:
            raise ValueError('identifier cannot be empty or None')

    def init(self):
        from superduperdb.base.document import Document
        from superduperdb.components.datatype import _BaseEncodable

        for f in dc.fields(self):
            item = getattr(self, f.name)
            if isinstance(item, Component):
                item.init()
            if isinstance(item, dict):
                setattr(self, f.name, Document(item).unpack(db=self.db))
            if isinstance(item, list):
                unpacked = Document({'_base': item}).unpack(db=self.db)
                setattr(self, f.name, unpacked)
            if isinstance(item, _BaseEncodable):
                item = item.unpack(db=self.db)
                setattr(self, f.name, item)

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
        return f'{self.type_id}/' f'{self.identifier}/' f'{self.version}'

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

    @staticmethod
    def import_(path: str):
        # Structure of directory is e.g:
        # extracted_archive
        # |_ compiled.json      // the recursive definitions of components
        # |_ 92392932839289     // the encododables as bytes (one file each)
        # |_ 22390232392832
        # |_ 23232932932832

        from superduperdb.components.datatype import serializers

        extracted_path = from_tarball(path)
        with open(os.path.join(extracted_path, 'compiled.json')) as f:
            references = json.load(f)

        for at in ['artifact', 'lazy_artifact']:
            for e in references[at]:
                with open(os.path.join(extracted_path, e), 'rb') as f:
                    r = references[at][e]
                    references[at][e] = serializers[
                        r['_content']['datatype']
                    ].decode_data(f.read())

        # This may get deprecated in favour of inline definitions (inside components)
        for s in references.get('serializable', {}):
            references['serializable'][s] = Serializable.decode(
                references['serializable'][s]
            )

        while True:
            # extract and replace the references inside the component defitions
            for c in references['component']:
                references['component'][c] = _replace_references(
                    references['component'][c], references
                )

            # Find the component definitions which don't refer to anything else
            # (leaf nodes)
            identifiers = [
                c
                for c in references['component']
                if not _find_references(references['component'][c])
                and isinstance(references['component'][c], dict)
            ]

            if not identifiers:
                raise Exception(
                    'Decompile due to unspecified references {_find_references}'
                )

            for c in identifiers:
                r = references['component'][c]
                references['component'][c] = Serializable.decode(r)

            previous_identifiers = identifiers[:]

            # Stop when all components have been converted
            if not [
                c
                for c in references['component']
                if isinstance(references['component'][c], dict)
            ]:
                break

        if len(previous_identifiers) == 1:
            return references['component'][previous_identifiers[0]]

        return [references['component'][c] for c in previous_identifiers]

    def export(self):
        from superduperdb.base.document import _encode_with_references

        references = defaultdict(dict)
        references['component'] = {self.unique_id: self}
        while True:
            keys = list(references['component'].keys())
            for k in keys[:]:
                v = references['component'][k]
                if isinstance(v, Component):
                    r = v.dict()
                    _encode_with_references(r, references)
                    references['component'][k] = dict(r)
            if all(isinstance(v, dict) for v in references['component'].values()):
                break
        with tempfile.TemporaryDirectory() as td:
            for at in ['artifact', 'lazy_artifact']:
                for k, v in references[at].items():
                    r = v.encode()
                    with open(f'{td}/{k}', 'wb') as f:
                        f.write(r['_content']['bytes'])
                    del r['_content']['bytes']
                    references[at][k] = r
            with open(f'{td}/compiled.json', 'w') as f:
                json.dump(references, f)
            to_tarball(td, self.identifier)
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
