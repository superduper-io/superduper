"""The component module provides the base class for all components in superduper.io."""

import dataclasses as dc
import json
import os
import shutil
import typing as t
from collections import OrderedDict, defaultdict
from datetime import datetime
from enum import Enum
from functools import wraps
from traceback import format_exc

import networkx

from superduper import logging
from superduper.base.annotations import trigger
from superduper.base.base import Base, BaseMeta
from superduper.base.constant import KEY_BLOBS, KEY_FILES, LENGTH_UUID
from superduper.base.status import (
    JOB_PHASE_FAILED,
    JOB_PHASE_PENDING,
    JOB_PHASE_RUNNING,
    init_status,
    running_status,
)
from superduper.misc.annotations import lazy_classproperty
from superduper.misc.importing import isreallyinstance
from superduper.misc.utils import hash_item

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer
    from superduper.base.metadata import Job


def _build_info_from_path(path: str):
    if os.path.exists(os.path.join(path, "component.json")):
        config = os.path.join(path, "component.json")
        with open(config) as f:
            config_object = json.load(f)
    elif os.path.exists(os.path.join(path, "component.yaml")):
        import yaml

        config = os.path.join(path, "component.yaml")
        with open(config) as f:
            config_object = yaml.safe_load(f.read())
    else:
        raise FileNotFoundError(
            f'`component.json` and `component.yaml` does not exist in the path {path}'
        )

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
            # assert len(sub_paths) == 1, f"Multiple files found in {file_id}"
            file_name = next(
                x for x in sub_paths if not x.startswith(".") or x.startswith("_")
            )
            files[file_id] = os.path.join(path, "files", file_id, file_name)
        config_object[KEY_FILES] = files

    return config_object


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


class Component(Base, metaclass=ComponentMeta):
    """Base class for all components in superduper.io.

    Class to represent superduper.io serializable entities
    that can be saved into a database.


    :param identifier: Identifier of the instance.
    :param upstream: A list of upstream components.

    :param db: Datalayer instance.
    """

    verbosity: t.ClassVar[int] = 0
    breaks: t.ClassVar[t.Sequence] = ()
    triggers: t.ClassVar[t.List] = []
    set_post_init: t.ClassVar[t.Sequence] = ('version', 'status')

    identifier: str
    upstream: t.Optional[t.List['Component']] = None

    db: dc.InitVar[t.Optional['Datalayer']] = None

    def __post_init__(self, db: t.Optional['Datalayer'] = None):
        self.db: Datalayer = db
        self.version: t.Optional[int] = None
        self.status: t.Dict = init_status()
        self.postinit()

    @property
    def uuid(self):
        """Get UUID."""
        t = self.get_merkle_tree(breaks=True)
        breaking = hash_item(
            [self.component, self.identifier] + [t[k] for k in self.breaks if k in t]
        )
        return breaking[:LENGTH_UUID]

    def get_merkle_tree(self, breaks: bool):
        """Get the merkle tree of the component.

        :param breaks: If set `true` only regard the parameters which break a version.
        """
        r = self._dict()
        s = self.class_schema
        keys = sorted(
            [k for k in r.keys() if k in s.fields and k not in {'uuid', 'status'}]
        )

        def get_hash(x):
            if breaks:
                return s.fields[x].uuid(r[x])[:32]
            else:
                return s.fields[x].hash(r[x])

        tree = OrderedDict(
            [(k, get_hash(k) if r[k] is not None else hash_item(None)) for k in keys]
        )
        return tree

    def diff(self, other: 'Component'):
        """Get the difference between two components.

        :param other: The other component to compare.
        """
        if not isreallyinstance(other, type(self)):
            raise ValueError('Cannot compare different types of components')

        if other.hash == self.hash:
            return {}

        m1 = self.get_merkle_tree(breaks=True)
        m2 = other.get_merkle_tree(breaks=True)
        d = []
        for k in m1:
            if m1[k] != m2[k]:
                d.append(k)
        return d

    @property
    def component(self):
        return self.__class__.__name__

    @lazy_classproperty
    def _new_fields(cls):
        """Get the schema of the class."""
        from superduper.misc.schema import get_schema

        s = get_schema(cls)[0]
        s['version'] = 'int'
        s['status'] = 'json'
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
                if only_initializing and r.status.phase != JOB_PHASE_PENDING:
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
        from superduper.base.datatype import Saveable

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

    def propagate_failure(self, exc: Exception):
        """Propagate the status of the component to its parents.

        :param exc: The exception to propagate.
        """
        self.db.metadata.set_component_status(
            component=self.component,
            uuid=self.uuid,
            status_update={
                'phase': JOB_PHASE_FAILED,
                'reason': str(exc),
                'message': format_exc(),
                'last_change_time': str(datetime.now()),
            },
        )

    @trigger('apply')
    def set_status(self):
        """Set the status of the component.

        :param status: The status to set the component to.
        """
        return self.db.metadata.set_component_status(
            self.__class__.__name__, self.uuid, status_update=running_status()
        )

    def create_jobs(
        self,
        context: str,
        event_type: str,
        ids: t.Sequence[str] | None = None,
        jobs: t.Sequence['Job'] = (),
        requires: t.Sequence[str] | None = None,
    ) -> t.List['Job']:
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
        local_jobs: t.List['Job'] = []
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
                            if r['status'] == JOB_PHASE_PENDING:
                                raise Exception(
                                    f'Component required component '
                                    f'{huuid} still initializing'
                                )
                            elif r['status'] == JOB_PHASE_RUNNING:
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
                    job: 'Job' = attr(job=True, context=context)
                else:
                    job: 'Job' = attr(ids=ids, job=True, context=context)  # type: ignore[no-redef]
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
            status_update: 'Job' = self.set_status(job=True, context=context)
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
        if not self.identifier:
            raise ValueError('identifier cannot be empty or None')

    def cleanup(self):
        """Method to clean the component."""
        pass

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

    def setup(self):
        """Method to help initiate component field dependencies."""

        def mro(item):
            objects = item.__class__.__mro__
            return [f'{o.__module__}.{o.__name__}' for o in objects]

        def _setup(item):

            if 'superduper.components.component.Component' in mro(item):
                item.setup()
                return item

            if isinstance(item, dict):
                return {k: _setup(i) for k, i in item.items()}

            if isinstance(item, list):
                return [_setup(i) for i in item]

            from superduper.base.datatype import Saveable

            if isinstance(item, Saveable):
                item.setup()
                return item.unpack()

            return item

        for f in dc.fields(self):
            item = getattr(self, f.name)
            item = _setup(item)
            setattr(self, f.name, item)

        return self

    def on_create(self):
        """Declare the component to the cluster."""
        assert self.db is not None

    def reload(self):
        """Reload the component from the datalayer."""
        assert self.db is not None

        latest_uuid = self.db.metadata.get_latest_uuid(self.component, self.identifier)
        if latest_uuid != self.uuid:
            return self.db.load(self.component, self.identifier)
        return self

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
        return Component.decode(config_object)

    def export(
        self,
        path: t.Optional[str] = None,
        defaults: bool = True,
        metadata: bool = False,
        format: str = "json",
    ):
        """
        Save `self` to a directory using super-duper protocol.

        :param path: Path to the directory to save the component.
        :param defaults: Whether to save default values.
        :param metadata: Whether to save metadata.
        :param format: Format to save the component. Accepts `json` and `yaml`.

        Created directory structure:
        ```
        |_<component>.json
        |_blobs/*
        |_files/*
        ```
        """
        if path is None:
            path = f"./{self.identifier}"

        if not os.path.exists(path):
            os.makedirs(path)

        # r = self.dict(defaults=defaults, metadata=metadata)
        # r = r.encode(defaults=defaults, metadata=metadata)
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

        if r.get(KEY_BLOBS):
            self._save_blobs_for_export(r[KEY_BLOBS], path)
            r.pop(KEY_BLOBS)

        if r.get(KEY_FILES):
            self._save_files_for_export(r[KEY_FILES], path)
            r.pop(KEY_FILES)

        if format == 'json':
            with open(os.path.join(path, "component.json"), "w") as f:
                json.dump(r, f, indent=2)
        elif format == 'yaml':
            import yaml

            with open(os.path.join(path, "component.yaml"), "w") as f:
                yaml.dump(r, f)
        else:
            raise ValueError(
                f"Format '{format}' not supported. "
                'Supported formats are:\n  - json\n  - yaml'
            )

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

    def _dict(self):
        return super().dict()

    def dict(self):
        """Get the dictionary representation of the component."""
        r = self._dict()
        r['version'] = self.version
        r['status'] = self.status
        r['uuid'] = self.uuid
        r['_path'] = self.__module__ + '.' + self.__class__.__name__
        return r

    @property
    def hash(self):
        t = self.get_merkle_tree(breaks=False)
        breaking_hashes = [t[k] for k in self.breaks if k in t]
        non_breaking_hashes = [t[k] for k in t if k not in self.breaks]
        breaking = hash_item(breaking_hashes)
        non_breaking = hash_item(non_breaking_hashes)
        return breaking[:32] + non_breaking[:32]


def ensure_setup(func):
    """Decorator to ensure that the model is initialized before calling the function.

    :param func: Decorator function.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "_is_setup", False):
            model_message = f"{self.__class__.__name__} : {self.identifier}"
            logging.debug(f"Initializing {model_message}")
            self.setup()
            self._is_setup = True
            logging.debug(f"Initialized  {model_message} successfully")
        return func(self, *args, **kwargs)

    return wrapper
