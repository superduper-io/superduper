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
from superduperdb.base.constant import KEY_BLOBS, KEY_BUILDS, KEY_FILES
from superduperdb.base.leaf import Leaf
from superduperdb.jobs.job import ComponentJob, Job

if t.TYPE_CHECKING:
    from superduperdb import Document
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.dataset import Dataset
    from superduperdb.components.datatype import DataType
    from superduperdb.components.plugin import Plugin


def _build_info_from_path(path: str):
    try:
        config = os.path.join(path, "component.json")
        with open(config) as f:
            config_object = json.load(f)
    except FileNotFoundError:
        try:
            config = os.path.join(path, "component.yaml")
            with open(config) as f:
                config_object = yaml.safe_load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError("No component.json or component.yaml found") from e

    config_object[KEY_BLOBS] = {}
    if os.path.exists(os.path.join(path, "blobs")):
        blobs = {}
        for file_id in os.listdir(os.path.join(path, "blobs")):
            with open(os.path.join(path, "blobs", file_id), "rb") as f:
                blobs[file_id] = f.read()
        config_object[KEY_BLOBS] = blobs

    config_object[KEY_FILES] = {}
    if os.path.exists(os.path.join(path, "files")):
        files = {}
        for file_id in os.listdir(os.path.join(path, "files")):
            sub_paths = os.listdir(os.path.join(path, "files", file_id))
            assert len(sub_paths) == 1, f"Multiple files found in {file_id}"
            file_name = sub_paths[0]
            files[file_id] = os.path.join(path, "files", file_id, file_name)
        config_object[KEY_FILES] = files

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
    r[KEY_BUILDS] = _build_leaves(r[KEY_BUILDS], db=db)[0]
    return r[KEY_BUILDS][r['_base']]


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


class Component(Leaf):
    """Base class for all components in SuperDuperDB.

    Class to represent SuperDuperDB serializable entities
    that can be saved into a database.

    :param artifacts: A dictionary of artifacts paths and `DataType` objects
    :param plugins: A list of plugins to be used in the component.
    """

    type_id: t.ClassVar[str] = 'component'
    leaf_type: t.ClassVar[str] = 'component'
    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, 'DataType']]] = ()
    set_post_init: t.ClassVar[t.Sequence] = ('version',)
    changed: t.ClassVar[set] = set([])
    plugins: t.Optional[t.List["Plugin"]] = None

    artifacts: dc.InitVar[t.Optional[t.Dict]] = None

    @property
    def children(self):
        """Get all the child components of the component."""
        r = self.dict().encode(leaves_to_keep=Leaf)
        out = [v for v in r['_builds'].values() if isinstance(v, Component)]
        out.extend(sum([c.children for c in out], []))
        return out

    @classmethod
    def from_template(
        self,
        identifier: str,
        template_body: t.Optional[t.Dict] = None,
        template_name: t.Optional[str] = None,
        db: t.Optional[Datalayer] = None,
        **kwargs,
    ):
        """Create a component from a template.

        :param identifier: Identifier of the component.
        :param template_body: Body of the template.
        :param template_name: Name of the template.
        :param db: Datalayer instance.
        """
        from superduperdb import Template

        if template_name:
            assert isinstance(db, Datalayer)
            template: Template = db.load('template', template_name)
        else:
            assert template_body is not None

            def _find_blobs(r, out):
                if isinstance(r, dict):
                    for v in r.values():
                        _find_blobs(v, out)
                if isinstance(r, list):
                    for i in r:
                        _find_blobs(i, out)
                if isinstance(r, str) and r.startswith('&:blob:'):
                    out.append(r.split(':')[-1])

            blobs: t.List[str] = []
            _find_blobs(template_body, blobs)
            template = Template(
                '_tmp',
                template=template_body,
                template_variables=list(kwargs.keys()),
                blobs=list(set(blobs)),
            )

        output = template(**kwargs)
        output.identifier = identifier
        return output

    @property
    def leaves(self):
        """Get all the leaves in the component."""
        r = self.dict()
        leaf_keys = [k for k in r.keys(True) if isinstance(r[k], Leaf)]
        return {k: r[k] for k in leaf_keys}

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
    def metadata(self):
        """Get metadata of the component."""
        return {
            'type_id': self.type_id,
            'version': self.version,
            'uuid': self.uuid,
        }

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

    @staticmethod
    def read(path: str, db: t.Optional[Datalayer] = None):
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
        was_zipped = False
        if path.endswith('.zip'):
            was_zipped = True
            import shutil

            shutil.unpack_archive(path)
            path = path.replace('.zip', '')

        config_object = _build_info_from_path(path=path)

        from superduperdb import Document

        if db is not None:
            for blob in os.listdir(path + '/' + 'blobs'):
                with open(path + '/blobs/' + blob, 'rb') as f:
                    bytes = f.read()
                    db.artifact_store.put_bytes(bytes, blob)

            out = Document.decode(config_object, db=db).unpack()
        else:

            def load_blob(blob):
                with open(path + '/blobs/' + blob, 'rb') as f:
                    return f.read()

            getters = {'blob': load_blob}

            out = Document.decode(config_object, getters=getters).unpack()
            if was_zipped:
                shutil.rmtree(path)
        return out

    def export(
        self,
        path: t.Optional[str] = None,
        format: str = "json",
        zip: bool = False,
        defaults: bool = False,
        metadata: bool = False,
        hr: bool = False,
    ):
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
        if path is None:
            path = f"./{self.identifier}"

        r = self.encode(defaults=defaults, metadata=metadata)

        def rewrite_keys(r, keys):
            if isinstance(r, dict):
                return {
                    rewrite_keys(k, keys): rewrite_keys(v, keys) for k, v in r.items()
                }
            if isinstance(r, list):
                return [rewrite_keys(i, keys) for i in r]
            if isinstance(r, str):
                for key in keys:
                    r = r.replace(key, keys[key])
            return r

        if hr:
            r = rewrite_keys(
                r, {k: f"blob_{i}" for i, k in enumerate(r.get(KEY_BLOBS))}
            )

        os.makedirs(path, exist_ok=True)
        if r.get(KEY_BLOBS):
            os.makedirs(os.path.join(path, "blobs"), exist_ok=True)
            for file_id, bytestr_ in r[KEY_BLOBS].items():
                filepath = os.path.join(path, "blobs", file_id)
                with open(filepath, "wb") as ff:
                    ff.write(bytestr_)

        r.pop(KEY_BLOBS)

        if r.get(KEY_FILES):
            os.makedirs(os.path.join(path, "files"), exist_ok=True)
            for file_id, file_path in r[KEY_FILES].items():
                file_path = file_path.rstrip("/")
                assert os.path.exists(file_path), f"File {file_path} not found"
                name = os.path.basename(file_path)
                save_path = os.path.join(path, "files", file_id, name)
                if os.path.isdir(file_path):
                    import shutil

                    shutil.copytree(file_path, save_path)
                else:
                    shutil.copy(file_path, save_path)

        r.pop(KEY_FILES)

        if format == "json":
            with open(os.path.join(path, "component.json"), "w") as f:
                json.dump(r, f, indent=2)

        elif format == "yaml":
            import re
            from io import StringIO

            from ruamel.yaml import YAML

            yaml = YAML()

            def custom_str_representer(dumper, data):
                if re.search(r"[^_a-zA-Z0-9 ]", data):
                    return dumper.represent_scalar(
                        "tag:yaml.org,2002:str", data, style='"'
                    )
                else:
                    return dumper.represent_scalar("tag:yaml.org,2002:str", data)

            yaml.representer.add_representer(str, custom_str_representer)

            stream = StringIO()
            yaml.dump(r, stream)

            output = str(stream.getvalue())

            with open(os.path.join(path, "component.yaml"), "w") as f:
                f.write(output)

        from superduperdb import REQUIRES

        with open(os.path.join(path, "requirements.txt"), "w") as f:
            f.write("\n".join(REQUIRES))

        if zip:
            self._zip_export(path)

    @staticmethod
    def _zip_export(path):
        import shutil

        name = path.split('/')[-1]
        os.makedirs(f'{path}/{name}', exist_ok=True)
        for file in [x for x in os.listdir(path) if x != name]:
            shutil.move(f'{path}/{file}', f'{path}/{name}/{file}')
        shutil.make_archive(path, 'zip', path)
        shutil.rmtree(path)

    def dict(self, metadata: bool = True, defaults: bool = True) -> 'Document':
        """A dictionary representation of the component."""
        from superduperdb import Document
        from superduperdb.components.datatype import Artifact, File

        r = super().dict(metadata=metadata, defaults=defaults)
        s = self.artifact_schema

        for k in s.fields:
            attr = getattr(self, k)
            if isinstance(attr, (Artifact, File)):
                r[k] = attr
            else:
                r[k] = s.fields[k](x=attr)  # artifact or file

        if metadata:
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

    def info(self, verbosity: int = 1):
        """Method to display the component information.

        :param verbosity: Verbosity level.
        """
        from superduperdb.misc.special_dicts import _display_component

        _display_component(self, verbosity=verbosity)


def ensure_initialized(func):
    """Decorator to ensure that the model is initialized before calling the function.

    :param func: Decorator function.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "_is_initialized", False):
            model_message = f"{self.__class__.__name__} : {self.identifier}"
            logging.debug(f"Initializing {model_message}")
            self.init()
            self._is_initialized = True
            logging.debug(f"Initialized  {model_message} successfully")
        return func(self, *args, **kwargs)

    return wrapper
