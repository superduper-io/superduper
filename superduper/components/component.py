"""The component module provides the base class for all components in superduper.io."""

import contextvars
import dataclasses as dc
import io
import json
import os
import pathlib
import shutil
import typing as t
from collections import OrderedDict, defaultdict
from contextlib import contextmanager, redirect_stdout
from functools import wraps

import networkx
import rich

from superduper import logging
from superduper.base.annotations import trigger
from superduper.base.base import Base, BaseMeta
from superduper.base.constant import KEY_BLOBS, KEY_FILES, LENGTH_UUID
from superduper.base.status import (
    STATUS_PENDING,
    STATUS_RUNNING,
    init_status,
)
from superduper.base.variables import _replace_variables
from superduper.misc.annotations import lazy_classproperty
from superduper.misc.importing import isreallyinstance
from superduper.misc.utils import hash_item

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer
    from superduper.base.metadata import Job


@contextmanager
def build_context(vars_dict: dict[str, t.Any] | None):
    """Context manager to set build variables for components.

    :param vars_dict: Dictionary of variables to set for the build context.
    """
    token1 = build_vars_var.set(vars_dict or {})
    token2 = context_swap.set({})
    try:
        yield
    finally:
        build_vars_var.reset(token1)
        context_swap.reset(token2)


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


def propagate_failure(f):
    """Propagate failure decorator.

    :param f: Function to decorate.
    """

    @wraps(f)
    def decorated(self, *args, **kwargs):
        try:
            return f(self, *args, **kwargs)
        except Exception as e:
            self.propagate_failure(e)
            raise e

    return decorated


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
        merged_metadata_fields = {}
        for base in bases:
            if hasattr(base, 'metadata_fields'):
                merged_metadata_fields.update(base.metadata_fields)
        new_cls.metadata_fields = {**new_cls.metadata_fields, **merged_metadata_fields}

        for base in bases:
            if hasattr(base, 'triggers'):
                new_cls.triggers.update(base.triggers)

        # Register new triggers from current class definition
        for attr_name, attr_value in dct.items():
            if hasattr(attr_value, 'events'):
                new_cls.triggers.add(attr_name)

        return new_cls


build_vars_var: contextvars.ContextVar[dict[str, t.Any]] = contextvars.ContextVar(
    "build_vars_var"
)
context_swap: contextvars.ContextVar[dict[str, t.Any]] = contextvars.ContextVar(
    "context_swap"
)


def current_build_vars(default: t.Any | None = None) -> dict[str, t.Any] | None:
    """Get the current build variables.

    :param default: Default value to return if no variables are set.
    """
    try:
        return build_vars_var.get()
    except LookupError:
        return default


class Component(Base, metaclass=ComponentMeta):
    """Base class for all components in superduper.io.

    Class to represent superduper.io serializable entities
    that can be saved into a database.


    :param identifier: Identifier of the instance.
    :param upstream: A list of upstream components.
    :param compute_kwargs: Keyword arguments to manage the compute environment.

    :param db: Datalayer instance.
    """

    verbosity: t.ClassVar[int] = 0
    breaks: t.ClassVar[t.Sequence] = ()
    triggers: t.ClassVar[t.List] = []
    services: t.ClassVar[t.List] = ()
    metadata_fields: t.ClassVar[t.Dict[str, t.Type]] = {
        'version': int,
        'status': str,
        'details': dict,
    }

    identifier: str
    upstream: t.Optional[t.List['Component']] = None
    compute_kwargs: t.Dict = dc.field(default_factory=dict)

    db: dc.InitVar[t.Optional['Datalayer']] = None

    def __post_init__(self, db: t.Optional['Datalayer'] = None):
        self.db: Datalayer = db

        self.version: t.Optional[int] = None
        self.status, self.details = init_status()

        self._original_parameters: t.Dict | None = None
        self._use_component_cache: bool = True

        self._handle_variables()
        self.postinit()

        assert self.identifier, "Identifier cannot be empty or None"

    def _build_tree(self, depth: int, tree=None):
        """Show the component."""
        if tree is None:
            from rich.tree import Tree

            tree = Tree(f"{self.huuid}")
        if depth == 0:
            return tree

        s = self.class_schema

        for k, v in self.dict(metadata=False).items():
            if k in {'_path', 'uuid', 'identifier'}:
                continue
            if isinstance(v, Component):
                subtree = tree.add(f"{k}: {v.huuid}")
                v._build_tree(depth - 1, subtree)
            elif str(s[k]) == 'ComponentList':
                if v:
                    subtree = tree.add(f"{k}")
                    for i, item in enumerate(v):
                        if isinstance(item, Component):
                            subsubtree = subtree.add(f"[{i}] {item.huuid}")
                            item._build_tree(depth - 1, subsubtree)
                        else:
                            subtree.add(f'[{i}] {item}')
            else:
                if v:
                    if isinstance(v, dict):
                        subtree = tree.add(k)
                        for sub_k, sub_v in v.items():
                            subtree.add(f"{sub_k}: {sub_v}")
                    else:
                        tree.add(f"{k}: {v}")
        return tree

    def _show_repr(self, depth: int = -1):
        with redirect_stdout(io.StringIO()) as buffer:
            self.show(depth=depth)
            return buffer.getvalue()

    def show(self, depth: int = -1):
        """Show the component in a tree format.

        :param depth: Depth of the tree to show.
        """
        tree_repr = self._build_tree(depth)
        rich.print(tree_repr)

    def save(self):
        """Save the component to the datalayer."""
        assert self.db is not None, "Datalayer is not set"
        self.db.apply(self, jobs=False, force=True)

    @property
    def metadata(self):
        """Get metadata of the component."""
        return {k: getattr(self, k) for k in self.metadata_fields}

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
        r = self._dict(metadata=False)
        s = self.class_schema
        keys = sorted(
            [
                k
                for k in r.keys()
                if k in s.fields and k not in {'uuid', 'status', 'details'}
            ]
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
            from superduper.base.datatype import ComponentRef

            if isinstance(r, (Component, ComponentRef)):
                if only_initializing and r.status.phase != STATUS_PENDING:
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
        from superduper.base.datatype import ComponentRef, Saveable

        r = self.dict().encode(leaves_to_keep=(Component, Saveable))
        out = [
            v.setup() or v
            for v in r['_builds'].values()
            if isinstance(v, (Component, ComponentRef))
        ]
        lookup: t.Dict[int, "Component"] = {}
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
        self.db.metadata.set_component_failed(
            component=self.component,
            uuid=self.uuid,
        )

    @classmethod
    def branch(cls):
        mro = cls.__mro__
        try:
            i = mro.index(Component)
        except ValueError:
            raise TypeError(f"{cls.__name__} is not a subclass of Component")
        # If cls is Component itself, there is no class 'down' from Component
        return None if i == 0 else mro[i - 1]

    @trigger('apply')
    def set_status(self):
        """Set the status of the component.

        :param status: The status to set the component to.
        """
        return self.db.metadata.set_component_status(
            self.__class__.__name__,
            self.uuid,
            status=STATUS_RUNNING,
            reason='The component is ready to use',
        )

    def create_table_events(self):
        """Create the table events for the component."""
        return {}

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
                            if r['status'] == STATUS_PENDING:
                                raise Exception(
                                    f'Component required component '
                                    f'{huuid} still initializing'
                                )
                            elif r['status'] == STATUS_RUNNING:
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

        if event_type == 'apply':  # and local_jobs:
            status_update: 'Job' = self.set_status(job=True, context=context)
            children_set = {(c.component, c.uuid) for c in self.get_children()}
            children_jobs = [
                job for job in jobs if (job.component, job.uuid) in children_set
            ]
            status_update.dependencies = [j.job_id for j in local_jobs + children_jobs]
            local_jobs.append(status_update)

        if self.compute_kwargs:
            for job in local_jobs:
                if job.method in self.compute_kwargs:
                    job.compute_kwargs = self.compute_kwargs[job.method]

        lookup: dict[str, "Job"] = {
            **{j.job_id: j for j in jobs},
            **{j.job_id: j for j in local_jobs},
        }
        for job in lookup.values():
            for dep in job.dependencies:
                dep_job = lookup[dep]
                if job.job_id not in dep_job.dependencies:
                    dep_job.inverse_dependencies.append(job.job_id)

        for job in lookup.values():
            job.inverse_dependencies = list(set(job.inverse_dependencies))
        return local_jobs

    @property
    def managed_tables(self):
        """Get all the managed tables in the component."""
        return []

    @property
    def leaves(self):
        """Get all the leaves in the component."""
        r = self.dict()
        leaf_keys = [k for k in r.keys(True) if isinstance(r[k], Base)]
        return {k: r[k] for k in leaf_keys}

    def _refresh(self, uuid_swaps: t.Optional[dict[str, str]] = None, **variables):
        def do_refresh(item):
            if isinstance(item, str):
                if '<var:' in item:
                    item = _replace_variables(item, uuid_swaps=uuid_swaps, **variables)
            if isinstance(item, list):
                for x in item:
                    do_refresh(x)
            if isinstance(item, dict):
                for v in item.values():
                    do_refresh(v)
            if isinstance(item, Component):
                if item._original_parameters is not None:
                    for k, v in item._original_parameters.items():
                        try:
                            setattr(item, k, v)
                        except AttributeError:
                            continue
                    item._original_parameters = None

                item._handle_variables()
                item._refresh(uuid_swaps=uuid_swaps, **variables)
            return item

        if '<var:' in self.identifier:
            self.identifier = _replace_variables(
                self.identifier, uuid_swaps=uuid_swaps, **variables
            )
        for k in self.class_schema.fields:
            v = do_refresh(getattr(self, k))
            try:
                setattr(self, k, v)
            except AttributeError:
                pass

    def _handle_variables(self):
        variables = build_vars_var.get(None)
        context_swap_value = context_swap.get({})

        if 'variables' in self.class_schema.fields and self.variables is not None:
            with build_context(self.variables):
                self._refresh(uuid_swaps=context_swap_value, **self.variables)
            return

        if not variables:
            return

        self._original_parameters = self.dict()

        former_uuid = self.uuid

        for f in dc.fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, (str, list, dict)):
                built = _replace_variables(
                    attr, uuid_swaps=context_swap_value, **variables
                )
                setattr(self, f.name, built)
            elif isinstance(attr, Base):
                built = _replace_variables(
                    attr, uuid_swaps=context_swap_value, **variables
                )
                setattr(self, f.name, built)

        if former_uuid != self.uuid:
            context_swap_value[former_uuid] = self.uuid

        context_swap.set(context_swap_value)

    def postinit(self):
        """Post initialization method."""

    def cleanup(self):
        """Method to clean the component."""
        # TODO deprecate in favour of dropping services and associated tables

        for service in self.services:
            getattr(self.db.cluster, service).drop_component(
                component=self.component, identifier=self.identifier
            )

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

            from superduper.base.datatype import ComponentRef, Saveable

            if isinstance(item, ComponentRef):
                item.component_cache = self._use_component_cache

            if isinstance(item, Saveable):
                item.setup()
                return item.unpack()

            return item

        for f in dc.fields(self):
            item = getattr(self, f.name)
            item = _setup(item)
            setattr(self, f.name, item)

        return self

    def reload(self):
        """Reload the component from the datalayer."""
        assert self.db is not None

        latest_uuid = self.db.metadata.get_latest_uuid(self.component, self.identifier)
        if latest_uuid != self.uuid:
            return self.db.load(self.component, self.identifier)
        return self

    @staticmethod
    def read(path: str, **variables) -> 'Component':
        """
        Read a `Component` instance from a directory created with `.export`.

        :param path: Path to the directory containing the component.
        :param variables: Variables to set on loading the component.

        Expected directory structure:
        ```
        |_component.json/yaml
        |_blobs/*
        |_files/*
        ```
        """

        @contextmanager
        def change_dir(destination):
            prev_dir = os.getcwd()
            os.chdir(destination)
            try:
                yield
            finally:
                os.chdir(prev_dir)

        pathlike = pathlib.Path(path)
        if not os.path.exists(str(pathlike / 'component.json')) and os.path.exists(
            str(pathlike / 'build.ipynb')
        ):
            with change_dir(path):
                import papermill

                papermill.execute_notebook(
                    './build.ipynb', '/tmp/build.ipynb', parameters={'APPLY': False}
                )

        config_object = _build_info_from_path(path=path)
        if variables and 'variables' not in config_object:
            raise ValueError(
                'This component does not support variables. '
                'Please omit **kwargs or re-export the component with `variables`.'
            )
        if variables:
            config_object['variables'].update(variables)
        return Component.decode(config_object)

    def export(
        self,
        path: t.Optional[str] = None,
        defaults: bool = False,
        format: str = "json",
    ):
        """
        Save `self` to a directory using super-duper protocol.

        :param path: Path to the directory to save the component.
        :param defaults: Whether to save default values.
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
        from superduper import CFG

        r = self.encode(
            defaults=defaults,
            metadata=False,
            keep_variables=True,
            export=True,
        )

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

    def _dict(self, metadata: bool = True):
        return super().dict(metadata=metadata)

    def dict(self, metadata: bool = True):
        """Get the dictionary representation of the component.

        :param metadata: If set `True` include metadata in the dictionary.
        """
        r = self._dict(metadata=metadata)
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

    def use_variables(self, **variables) -> 'Component':
        """Use variables in the component.

        :param variables: Variables to use in the component.
        """
        if not hasattr(self, 'variables'):
            raise ValueError(
                'This component does not support variables. '
                'Please omit **kwargs or re-export the component with `variables`.'
            )
        r = self.encode(
            defaults=False,
            metadata=False,
            keep_variables=True,
        )
        r['variables'].update(variables)
        return self.decode(r, db=self.db)  # type: ignore[arg-type]
