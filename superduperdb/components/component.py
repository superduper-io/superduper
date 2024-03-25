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

            if isinstance(item, _BaseEncodable):
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
    def _import_from_references(references, db: t.Optional[Datalayer] = None):
        if db is not None:
            all_references = _find_references(references['component'])
            for k in all_references:
                # E.g. `$component/<type_id>/<identifier>/<version>`
                split = k.split('/')
                if split[0][1:] != 'component':
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
                    references['component'][k] = db.load(type_id, identifier, version)
                    logging.info(f'Loaded {k} from db')
                except Exception as e:
                    logging.warn(f'Failed to load {k} from db: {e}')

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
                    f'Decompile failed due to unspecified references {_find_references}'
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

        from superduperdb.components.stack import Stack

        return Stack(
            'stack',
            components=[references['component'][c] for c in previous_identifiers],
        )

    @staticmethod
    def import_from_references(definition: t.Dict, db: Datalayer):
        from superduperdb import Document
        from superduperdb.components.datatype import LazyArtifact

        if definition.get('lazy_artifact'):
            for a in definition['lazy_artifact']:
                definition['lazy_artifact'][a] = LazyArtifact(
                    **definition['lazy_artifact'][a]['_content']
                )
            definition['lazy_artifact'] = dict(
                Document.decode(definition['lazy_artifact'], db=db)
            )
        if definition.get('artifact'):
            definition['artifact'] = dict(
                Document.decode(definition['artifact'], db=db)
            )
        return Component._import_from_references(definition, db=db)

    @staticmethod
    def import_from_path(
        path: str,
    ):
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

        return Component._import_from_references(references)

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
            references['component'] = {self.unique_id: self}
        except Exception:
            references['component'] = {f'{self.type_id}/{self.identifier}': self}
        while True:
            keys = list(references['component'].keys())[:]
            for k in keys:
                if isinstance(references['component'][k], Component):
                    references['component'][k] = process_subpart(
                        references['component'][k], references
                    )
            if all(isinstance(v, dict) for v in references['component'].values()):
                break
        return references

    def export_to_path(self):
        references = self.export_to_references()
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


if __name__ == '__main__':
    from superduperdb import ObjectModel, superduper
    from superduperdb.components.stack import Stack
    from superduperdb.ext.numpy import array

    db = superduper('mongomock://test', artifact_store='filesystem:///tmp/artifacts')

    m = Stack(
        'test_stack',
        components=[
            ObjectModel(
                'test',
                object=lambda x: x + 2,
                datatype=array('float32', shape=(32,)),
            ),
            ObjectModel(
                'test2',
                object=lambda x: x + 3,
                datatype=array('float32', shape=(16,)),
            ),
        ],
    )

    db.add(m)

    r = m.export_to_path()

    for k in r['lazy_artifact']:
        r['lazy_artifact'][k] = {
            '_content': {
                'file_id': r['lazy_artifact'][k]['_content']['file_id'],
                'datatype': db.datatypes[r['lazy_artifact'][k]['_content']['datatype']],
            }
        }

    pprint.pprint(r)

    component = m.import_from_references(r, db)
    component.db = db

    pprint.pprint(component)

    component.init()

    pprint.pprint(component)
