"""The component module provides the base class for all components in superduper.io."""

# TODO: why do I need this? Aren't we already in the future?
from __future__ import annotations

import dataclasses as dc
import json
import os
import shutil
import typing as t
from collections import defaultdict, namedtuple
from enum import Enum
from functools import wraps
import uuid

import networkx
import yaml

from superduper import logging
from superduper.base.annotations import trigger
from superduper.base.constant import KEY_BLOBS, KEY_FILES
from superduper.base.event import Job
from superduper.base.base import Base, BaseMeta
from superduper.misc.annotations import lazy_classproperty

if t.TYPE_CHECKING:
    from superduper import Document
    from superduper.base.datalayer import Datalayer
    from superduper.components.plugin import Plugin


class Status(str, Enum):
    """Status enum.

    # noqa
    """

    initializing = 'initializing'
    ready = 'ready'


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


ComponentTuple = namedtuple('ComponentTuple', ['type_id', 'identifier', 'version'])
ComponentTuple.__doc__ = 'noqa'


def _is_optional_callable(annotation) -> bool:
    """Tell if an annotation is t.Optional[t.Callable].

    >>> is_optional_callable(t.Optional[t.Callable])
    True
    """
    # Check if the annotation is of the form Optional[...]
    if t.get_origin(annotation) is t.Union:
        # Get the type inside Optional and check if it is Callable
        inner_type = t.get_args(annotation)[0]  # Optional[X] means X is at index 0
        return inner_type is t.Callable
    return False


class ComponentMeta(BaseMeta):
    """Metaclass for the `Component` class.

    This component is used to aggregate all the triggers
    from the base classes. # noqa
    """

    def __new__(cls, name, bases, dct):
        """Create a new class.

        # noqa
        """
        # Create the new class using type.__new__
        new_cls = super().__new__(cls, name, bases, dct)
        # Initialize the trigger registry
        new_cls.triggers = set()

        for base in bases:
            if hasattr(base, 'triggers'):
                new_cls.triggers.update(base.triggers)

        # Register new triggers from current class definition
        for attr_name, attr_value in dct.items():
            if hasattr(attr_value, 'events'):
                new_cls.triggers.add(attr_name)

        return new_cls


def build_uuid():
    """Build UUID."""
    return str(uuid.uuid4()).replace('-', '')[:16]


class Component(Base, metaclass=ComponentMeta):
    """Base class for all components in superduper.io.

    Class to represent superduper.io serializable entities
    that can be saved into a database.


    :param identifier: Identifier of the instance.
    :param uuid: UUID of the instance.
    :param upstream: A list of upstream components.
    :param cache: (Optional) If set `true` the component will not be cached
                  during primary job of the component i.e on a distributed
                  cluster this component will be reloaded on every component
                  task e.g model prediction.
    :param status: What part of the lifecycle the component is in.
    :param build_variables: Variables which were supplied to a template to build.
    :param build_template: Template which was used to build.
    :param db: Datalayer instance.
    """

    breaks: t.ClassVar[t.Sequence] = ()
    triggers: t.ClassVar[t.List] = []
    set_post_init: t.ClassVar[t.Sequence] = ('version',)

    identifier: str
    uuid: str = dc.field(default_factory=build_uuid)
    upstream: t.Optional[t.List['Component']] = None
    cache: t.Optional[bool] = True
    status: t.Optional[Status] = None
    build_variables: t.Dict | None = None
    build_template: str | None = None

    db: dc.InitVar[t.Optional['Datalayer']] = None

    def __post_init__(self, db: t.Optional['Datalayer'] = None):
        self.db = db
        self.postinit()

    @property
    def component(self):
        return self.__class__.__name__

    @lazy_classproperty
    def _new_fields(cls):
        """Get the schema of the class."""
        from superduper.misc.schema import get_schema

        s, a = get_schema(cls)
        s['version'] = 'int'
        return s

    @staticmethod
    def sort_components(components):
        """Sort components based on topological order.

        :param components: List of components.
        """
        logging.info('Resorting components based on topological order.')
        G = networkx.DiGraph()
        lookup = {c.huuid: c for c in components}
        for k in lookup:
            G.add_node(k)
            for d in lookup[k].get_children_refs():  # dependencies:
                if d in lookup:
                    G.add_edge(d, k)

        nodes = list(networkx.topological_sort(G))
        logging.info(f'New order of components: {nodes}')
        return [lookup[n] for n in nodes]

    @property
    def huuid(self):
        """Return a human-readable uuid."""
        return f'{self.__class__.__name__}:{self.identifier}:{self.uuid}'

    def handle_update_or_same(self, other: 'Component'):
        """Handle when a component is changed without breaking changes.

        :param other: The other component to handle.
        """
        other.uuid = self.uuid
        other.version = self.version

    def get_children_refs(self, deep: bool = False, only_initializing: bool = False):
        """Get all the children of the component.

        :param deep: If set `true` get all the children of the component.
        :param only_initializing: If set `true` get only the initializing children.
        """
        r = self.dict()

        def _find_refs(r):
            if (
                isinstance(r, str)
                and r.startswith('&:component:')
                and not only_initializing
            ):
                return [':'.join(r.split(':')[2:])]
            if isinstance(r, dict):
                return sum([_find_refs(v) for v in r.values()], [])
            if isinstance(r, list):
                return sum([_find_refs(x) for x in r], [])
            if isinstance(r, Component):
                if only_initializing and r.status != Status.initializing:
                    return []
                else:
                    return [r.huuid]
            return []

        out = _find_refs(r)
        return sorted(list(set(out)))

    def get_children(self, deep: bool = False) -> t.List["Component"]:
        """Get all the children of the component.

        :param deep: If set `True` get all recursively.
        """
        from superduper.components.datatype import Saveable

        r = self.dict().encode(leaves_to_keep=(Component, Saveable))
        out = [v for v in r['_builds'].values() if isinstance(v, Component)]
        lookup = {}
        for v in out:
            lookup[id(v)] = v
        if deep:
            children = list(lookup.values())
            for v in children:
                sub = v.get_children(deep=True)
                for s in sub:
                    lookup[id(s)] = s
        return list(lookup.values())

    @property
    def children(self):
        """Get all the child components of the component."""
        return self.get_children(deep=False)

    def _filter_trigger(self, name, event_type):
        attr = getattr(self, name)
        if event_type not in getattr(attr, 'events', ()):
            return False

        for item in attr.requires:
            if getattr(self, item) is None:
                return False

        return True

    def get_triggers(self, event_type, requires: t.Sequence[str] | None = None):
        """
        Get all the triggers for the component.

        :param event_type: event_type
        :param requires: the methods which should run first
        """
        # Get all of the methods in the class which have the `@trigger` decorator
        # and which match the event type
        triggers = [
            attr_name
            for attr_name in self.triggers
            if self._filter_trigger(attr_name, event_type)
        ]
        if requires:
            triggers = [
                t
                for t in triggers
                if set(getattr(self, t).requires).intersection(requires)
            ]
        return triggers

    @trigger('apply')
    def set_status(self, status: Status):
        """Set the status of the component.

        :param status: The status to set the component to.
        """
        return self.db.metadata.set_component_status(
            self.__class__.__name__, self.uuid, str(status)
        )

    def create_jobs(
        self,
        context: str,
        event_type: str,
        ids: t.Sequence[str] | None = None,
        jobs: t.Sequence[Job] = (),
        requires: t.Sequence[str] | None = None,
    ) -> t.List[Job]:
        """Deploy apply jobs for the component.

        :param context: The context of the component.
        :param event_type: The event type.
        :param ids: The ids of the component.
        :param jobs: The jobs of the component.
        :param requires: The requirements of the component.
        """
        # TODO replace this with a DAG check
        max_it = 100
        it = 0

        triggers = self.get_triggers(event_type, requires=requires)
        triggers = list(set(triggers) - {'set_status'})

        # local_job_lookup is {j.method_name: j.job_id for j in local_jobs}
        local_job_lookup = {}
        local_jobs: t.List[Job] = []
        # component_to_job_lookup is {c.uuid: <list-of-job-ids-for-component>}
        component_to_job_lookup = defaultdict(list)
        for j in jobs:
            component_to_job_lookup[j.uuid].append(j.job_id)

        while triggers:
            for attr_name in triggers:
                attr = getattr(self, attr_name)

                depends = [d for d in attr.depends if d in triggers]

                # Check that the dependencies of the job are ready
                # If not skip until the next round
                if not all(f in jobs for f in depends):
                    continue

                # Dependencies come from these local jobs
                # plus the jobs from the child components
                inter_component_dependencies = []  # list of job-ids
                for child in self.get_children_refs():
                    child_uuid = child.split(':')[-1]
                    if child_uuid not in component_to_job_lookup:
                        continue
                    try:
                        inter_component_dependencies.extend(
                            component_to_job_lookup[child_uuid]
                        )
                    except KeyError as e:
                        r = self.db.metadata.get_component_by_uuid(uuid=child_uuid)
                        class_name = r['_path'].split('.')[-1]
                        huuid = f'{class_name}:{r["identifier"]}:{r["uuid"]}'
                        if event_type == 'apply':
                            if r['status'] == 'initializing':
                                raise Exception(
                                    f'Component required component '
                                    f'{huuid} still initializing'
                                )
                            elif r['status'] == 'ready':
                                logging.info(
                                    f'Detected a ready component ' f'dependency {huuid}'
                                )
                        else:
                            if r.get('cdc_table') is not None:
                                raise Exception(
                                    f"Missing an upstream dependency "
                                    f"{huuid} on table {r['cdc_table']}"
                                ) from e

                # TODO this seems to suggest that local_jobs is a dictionary
                intra_component_dependencies = [
                    local_jobs[f] for f in depends if f in local_jobs
                ]
                dependencies = (
                    inter_component_dependencies + intra_component_dependencies
                )

                # 'apply' event doesn't need inputs
                if event_type == 'apply':
                    job: Job = attr(job=True, context=context)
                else:
                    job: Job = attr(ids=ids, job=True, context=context)  # type: ignore[no-redef]
                job.dependencies = dependencies

                # TODO rename the call method to execute or deploy
                # This adds the promised output "future" to the saved outputs
                # out_futures[attr_name] = job(dependencies=dependencies)
                local_job_lookup[attr_name] = job
                local_jobs.append(job)

                triggers = [t for t in triggers if t != attr_name]

            it += 1
            if it > max_it:
                raise Exception('Circular job dependency detected')
            if len(local_jobs) == len(triggers):
                break

        if event_type == 'apply' and local_jobs:
            status_update: Job = self.set_status(
                status=Status.ready, job=True, context=context
            )
            status_update.dependencies = [j.job_id for j in local_jobs]
            local_jobs.append(status_update)
        return local_jobs

    @property
    def leaves(self):
        """Get all the leaves in the component."""
        r = self.dict()
        leaf_keys = [k for k in r.keys(True) if isinstance(r[k], Base)]
        return {k: r[k] for k in leaf_keys}

    def postinit(self):
        """Post initialization method."""
        self.version: t.Optional[int] = None
        if not self.identifier:
            raise ValueError('identifier cannot be empty or None')

    def cleanup(self, db: Datalayer):
        """Method to clean the component.

        :param db: The `Datalayer` to use for the operation.
        """
        db.cluster.cache.drop(self)

    def _get_metadata(self):
        """Get metadata of the component."""
        metadata = {
            'version': self.version,
            'status': self.status,
        }
        return metadata

    @property
    def dependencies(self):
        """Get dependencies on the component."""
        return ()

    # TODO why both methods?
    def init(self):
        """Method to help initiate component field dependencies."""
        self.unpack()

    def unpack(self):
        """Method to unpack the component.

        This method is used to initialize all the fields of the component and leaf
        """

        def _init(item):
            if isinstance(item, Component):
                item.init()
                return item

            if isinstance(item, dict):
                return {k: _init(i) for k, i in item.items()}

            if isinstance(item, list):
                return [_init(i) for i in item]

            from superduper.components.datatype import Saveable

            if isinstance(item, Saveable):
                item.init()
                return item.unpack()

            return item

        for f in dc.fields(self):
            item = getattr(self, f.name)
            item = _init(item)
            setattr(self, f.name, item)

        return self

    def _pre_create(self, db: Datalayer, startup_cache: t.Dict = {}):
        self.status = Status.initializing

    def pre_create(self, db: Datalayer):
        """Called the first time this component is created.

        :param db: the db that creates the component.
        """
        assert db
        for child in self.get_children():
            child.pre_create(db)
        self._pre_create(db)

    def on_create(self, db: Datalayer) -> None:
        """Called after the first time this component is created.

        Generally used if ``self.version`` is important in this logic.

        :param db: the db that creates the component.
        """
        assert db
        self.declare_component(db.cluster)

    def declare_component(self, cluster):
        """Declare the component to the cluster.

        :param cluster: The cluster to declare the component to.
        """
        if self.cache:
            logging.info(f'Adding {self.component}:{self.identifier} to cache')
            cluster.cache.put(self)

    @staticmethod
    def read(path: str, db: t.Optional[Datalayer] = None):
        """
        Read a `Component` instance from a directory created with `.export`.

        :param path: Path to the directory containing the component.
        :param db: Datalayer instance to be used to read the component.

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

        from superduper import Document

        if db is not None:
            for blob in os.listdir(path + '/' + 'blobs'):
                with open(path + '/blobs/' + blob, 'rb') as f:
                    data = f.read()
                    db.artifact_store.put_bytes(data, blob)

            out = Document.decode(config_object, db=db).unpack()
        else:
            from superduper.backends.local.artifacts import FileSystemArtifactStore

            artifact_store = FileSystemArtifactStore(
                conn=path,
                name='tmp_artifact_store',
                files='files',
                blobs='blobs',
            )
            db = namedtuple('tmp_db', field_names=('artifact_store',))(
                artifact_store=artifact_store
            )
            out = Document.decode(config_object, db=db).unpack()
            if was_zipped:
                shutil.rmtree(path)
        return out

    def export(
        self,
        path: t.Optional[str] = None,
        format: str = "json",
        zip: bool = False,
        defaults: bool = True,
        metadata: bool = False,
        hr: bool = False,
        component: str = 'component',
    ):
        """
        Save `self` to a directory using super-duper protocol.

        :param path: Path to the directory to save the component.
        :param format: Format to save the component in (json/ yaml).
        :param zip: Whether to zip the directory.
        :param defaults: Whether to save default values.
        :param metadata: Whether to save metadata.
        :param hr: Whether to save human-readable blobs.
        :param component: Name of the component file.

        Created directory structure:
        ```
        |_<component>.json/yaml
        |_blobs/*
        |_files/*
        ```
        """
        if path is None:
            path = f"./{self.identifier}"

        if not os.path.exists(path):
            os.makedirs(path)

        r = self.dict(defaults=defaults, metadata=metadata)
        r = r.encode(defaults=defaults, metadata=metadata)

        if not metadata:
            del r['uuid']

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
            blobs = r.get(KEY_BLOBS, {})
            r = rewrite_keys(r, {k: f"blob_{i}" for i, k in enumerate(blobs)})

        if r.get(KEY_BLOBS):
            self._save_blobs_for_export(r[KEY_BLOBS], path)
            r.pop(KEY_BLOBS)

        if r.get(KEY_FILES):
            self._save_files_for_export(r[KEY_FILES], path)
            r.pop(KEY_FILES)

        if format == "json":
            with open(os.path.join(path, f"{component}.json"), "w") as f:
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

            with open(os.path.join(path, f"{component}.yaml"), "w") as f:
                f.write(output)

        # No longer relevant with plugins
        # Needs to be created manually currently
        # with open(os.path.join(path, "requirements.txt"), "w") as f:
        #     f.write("\n".join(REQUIRES))

        if zip:
            self._zip_export(path)

    @staticmethod
    def _save_blobs_for_export(blobs, path):
        if not blobs:
            return
        os.makedirs(os.path.join(path, "blobs"), exist_ok=True)
        for file_id, bytestr_ in blobs.items():
            filepath = os.path.join(path, "blobs", file_id)
            with open(filepath, "wb") as ff:
                ff.write(bytestr_)

    @staticmethod
    def _save_files_for_export(files, path):
        if not files:
            return
        os.makedirs(os.path.join(path, "files"), exist_ok=True)
        for file_id, file_path in files.items():
            file_path = file_path.rstrip("/")
            assert os.path.exists(file_path), f"File {file_path} not found"
            name = os.path.basename(file_path)
            save_path = os.path.join(path, "files", file_id, name)
            os.makedirs(os.path.join(path, "files", file_id), exist_ok=True)
            if os.path.isdir(file_path):
                if os.path.exists(save_path):
                    shutil.rmtree(save_path)
                shutil.copytree(file_path, save_path)
            else:
                shutil.copy(file_path, save_path)

    @staticmethod
    def _zip_export(path):
        import shutil

        name = path.split('/')[-1]
        os.makedirs(f'{path}/{name}', exist_ok=True)
        for file in [x for x in os.listdir(path) if x != name]:
            shutil.move(f'{path}/{file}', f'{path}/{name}/{file}')
        shutil.make_archive(path, 'zip', path)
        shutil.rmtree(path)

    def dict(
        self,
        metadata: bool = True,
        defaults: bool = True,
        refs: bool = False,
        path: bool = True,
    ) -> 'Document':
        """A dictionary representation of the component.

        :param metadata: If set `true` include metadata.
        :param defaults: If set `true` include defaults.
        :param refs: If set `true` include references.
        :param path: If `true` include path.
        """
        from superduper import Document

        r = super().dict(metadata=metadata, defaults=defaults, path=path)

        def _convert_components_to_refs(r):
            if isinstance(r, dict):
                return {k: _convert_components_to_refs(v) for k, v in r.items()}
            if isinstance(r, list):
                return [_convert_components_to_refs(x) for x in r]
            if isinstance(r, Component):
                return f'&:component:{r.huuid}'
            return r

        if refs:
            r = Document(_convert_components_to_refs(r), schema=self.class_schema)

        if metadata:
            r['version'] = self.version
            r['identifier'] = self.identifier

        if r.get('status') is not None:
            r['status'] = str(self.status)

        return r

    def info(self, verbosity: int = 1):
        """Method to display the component information.

        :param verbosity: Verbosity level.
        """
        from superduper.misc.special_dicts import _display_component

        _display_component(self, verbosity=verbosity)

    @property
    def cdc_table(self):
        """Get table for cdc."""
        return False


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