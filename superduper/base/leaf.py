import dataclasses as dc
import importlib
import inspect
import typing as t
import uuid

from superduper.base.constant import KEY_BLOBS, KEY_BUILDS, KEY_FILES
from superduper.misc.annotations import (
    extract_parameters,
    lazy_classproperty,
    replace_parameters,
)
from superduper.misc.serialization import asdict

_CLASS_REGISTRY = {}

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


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


class LeafMeta(type):
    """Metaclass that merges docstrings # noqa."""

    def __new__(mcs, name, bases, namespace):
        """Create a new class with merged docstrings # noqa."""
        # Prepare namespace by extracting annotations and handling fields
        annotations = namespace.get('__annotations__', {})
        for k, v in list(namespace.items()):
            if isinstance(v, (type, dc.InitVar)):
                annotations[k] = v
            if isinstance(v, dc.Field):
                if v.type is not None:
                    annotations[k] = v.type

        # Update namespace with proper annotations
        namespace['__annotations__'] = annotations

        # Determine if any bases are dataclasses and
        # apply the appropriate dataclass decorator
        #
        dataclass_params = namespace.get('_dataclass_params', {}).copy()
        if bases and any(dc.is_dataclass(b) for b in bases):
            dataclass_params['kw_only'] = True
            dataclass_params['repr'] = not name.endswith('Query')
            # Derived classes: kw_only=True
        else:
            # Base class: kw_only=False
            dataclass_params['kw_only'] = False

        cls = dc.dataclass(**dataclass_params)(
            super().__new__(mcs, name, bases, namespace)
        )

        # Merge docstrings from parent classes
        parent_doc = next(
            (parent.__doc__ for parent in inspect.getmro(cls)[1:] if parent.__doc__),
            None,
        )
        if parent_doc:
            parent_params = extract_parameters(parent_doc)
            child_doc = cls.__doc__ or ''
            child_params = extract_parameters(child_doc)
            for k in child_params:
                parent_params[k] = child_params[k]
            placeholder_doc = replace_parameters(child_doc)
            param_string = ''
            for k, v in parent_params.items():
                v = '\n    '.join(v)
                param_string += f':param {k}: {v}\n'
            cls.__doc__ = placeholder_doc.replace('!!!', param_string)

        return cls


def build_uuid():
    """Build UUID."""
    return str(uuid.uuid4()).replace('-', '')[:16]


class Leaf(metaclass=LeafMeta):
    """Base class for all leaf classes.

    :param identifier: Identifier of the leaf.
    :param db: Datalayer instance.
    :param uuid: UUID of the leaf.
    """

    set_post_init: t.ClassVar[t.Sequence[str]] = ()

    identifier: str
    db: dc.InitVar[t.Optional['Datalayer']] = None
    uuid: str = dc.field(default_factory=build_uuid)

    @lazy_classproperty
    def _new_fields(cls):
        """Get the schema of the class."""
        from superduper.misc.schema import get_schema

        s, a = get_schema(cls)
        return s

    @lazy_classproperty
    def class_schema(cls):
        fields = {}
        from superduper.components.datatype import INBUILT_DATATYPES
        from superduper.components.schema import Schema

        named_fields = cls._new_fields
        for f in named_fields:
            fields[f] = INBUILT_DATATYPES[named_fields[f]]
        out = Schema(fields=fields)
        return out

    @staticmethod
    def get_cls_from_path(path):
        """Get class from a path.

        :param path: Import path to the class.
        """
        parts = path.split('.')
        cls = parts[-1]
        module = '.'.join(parts[:-1])
        module = importlib.import_module(module)
        return getattr(module, cls)

    @staticmethod
    def get_cls_from_blob(blob_ref, db):
        """Get class from a blob reference.

        :param blob_ref: Blob reference identifier.
        :param db: Datalayer instance.
        """
        from superduper.components.datatype import DEFAULT_SERIALIZER, Blob

        bytes_ = Blob(identifier=blob_ref.split(':')[-1], db=db).unpack()
        return DEFAULT_SERIALIZER.decode_data(bytes_)

    def reconnect(self, db):
        """Reconnect the object to a new datalayer.

        :param db: Datalayer instance.
        """
        r = self.dict()
        if '_path' in r:
            r.pop('_path')
        elif '_object' in r:
            r.pop('_object')
        return self.from_dict(r, db=db)

    @classmethod
    def from_dict(cls, r: t.Dict, db: 'Datalayer'):
        try:
            out = cls(**{k: v for k, v in r.items() if k != '_path'}, db=db)
            return out
        except TypeError as e:
            if 'got an unexpected keyword argument' in str(e):
                init_params = {
                    k: v
                    for k, v in r.items()
                    if k in inspect.signature(cls.__init__).parameters
                }
                post_init_params = {
                    k: v for k, v in r.items() if k in cls.set_post_init
                }
                instance = cls(**init_params, db=db)
                for k, v in post_init_params.items():
                    setattr(instance, k, v)
                return instance
            raise e

    def postinit(self):
        """Post-initialization method."""
        pass

    def _get_metadata(self):
        return {}

    @property
    def metadata(self):
        """Get metadata of the object."""
        return self._get_metadata()

    def __post_init__(self, db: t.Optional['Datalayer'] = None):
        self.db = db
        self.postinit()

    @property
    def leaves(self):
        """Get all leaves in the object."""
        return {
            f.name: getattr(self, f.name)
            for f in dc.fields(self)
            if isinstance(getattr(self, f.name), Leaf)
        }

    @classmethod
    def cls_encode(cls, item: 'Leaf', builds, blobs, files, leaves_to_keep=()):
        """Encode a dictionary component into a `Component` instance.

        :param r: Object to be encoded.
        """
        return item.encode(
            builds=builds, blobs=blobs, files=files, leaves_to_keep=leaves_to_keep
        )

    # TODO signature is inverted from `Component.encode`
    def encode(
        self,
        leaves_to_keep=(),
        metadata: bool = True,
        defaults: bool = True,
        builds: t.Dict = None,
        blobs: t.Dict = None,
        files: t.Dict = None,
    ):
        """Encode itself.

        After encoding everything is a vanilla dictionary (JSON + bytes).

        :param leaves_to_keep: Leaves to keep.
        :param metadata: Include metadata.
        :param defaults: Include default values.
        :param builds: Builds.
        :param blobs: Blobs.
        :param files: Files.
        """
        builds: t.Dict = {}
        blobs: t.Dict = {}
        files: t.Dict = {}
        uuids_to_keys: t.Dict = {}

        r = self.dict(metadata=metadata, defaults=defaults)
        r = self.class_schema.encode_data(r, builds=builds, blobs=blobs, files=files)

        def _replace_loads_with_references(record, lookup):
            if isinstance(record, str) and record.startswith('&:component:'):
                uuid = record.split(':')[-1]
                key = lookup[uuid]
                return f'?{key}'
            if isinstance(record, list):
                return [_replace_loads_with_references(x, lookup) for x in record]
            if isinstance(record, dict):
                return {
                    k: _replace_loads_with_references(v, lookup)
                    for k, v in record.items()
                }
            return record

        lookup = {v['uuid']: k for k, v in builds.items() if 'uuid' in v}
        builds = _replace_loads_with_references(builds, lookup)

        def _replace_uuids_with_keys(record):
            import json

            dump = json.dumps(record)
            for k, v in uuids_to_keys.items():
                dump = dump.replace(v, f'?({k}.uuid)')
            return json.loads(dump)

        if not metadata:
            builds = _replace_uuids_with_keys(builds)
            for v in builds.values():
                if 'uuid' in v:
                    del v['uuid']
            if 'uuid' in r:
                del r['uuid']

        # TODO deprecate this wrapper (not needed)
        return {
            **r,
            KEY_BUILDS: builds,
            KEY_BLOBS: blobs,
            KEY_FILES: files,
        }

    def set_variables(self, db: t.Union['Datalayer', None] = None, **kwargs) -> 'Leaf':
        """Set free variables of self.

        :param db: Datalayer instance.
        :param kwargs: Keyword arguments to pass to `_replace_variables`.
        """
        from superduper import Document
        from superduper.base.variables import _replace_variables

        r = self.encode()
        rr = _replace_variables(r, **kwargs)
        return Document.decode(rr, db=db).unpack()

    @property
    def variables(self) -> t.List[str]:
        """Get list of variables in the object."""
        from superduper.base.variables import _find_variables

        return list(set(_find_variables(self.encode())))

    # TODO this is buggy - defaults don't work
    @property
    def defaults(self):
        """Get the default parameter values."""
        out = {}
        fields = dc.fields(self)
        for f in fields:
            value = getattr(self, f.name)
            if f.default is not dc.MISSING and f.default and value == f.default:
                out[f.name] = value
            elif (
                f.default_factory is not dc.MISSING
                and f.default
                and value == f.default_factory()
            ):
                out[f.name] = value
        return out

    # TODO the signature does not agree with the `Component.dict` method
    def dict(
        self,
        metadata: bool = True,
        defaults: bool = True,
        schema: bool = False,
        path: bool = True,
    ):
        """Return dictionary representation of the object.

        :param metadata: Include metadata.
        :param defaults: Include default values.
        :param schema: Include schema.
        :param path: Include path.
        """
        from superduper import Document

        r = asdict(self)

        if not defaults:
            for k, v in self.defaults.items():
                if k in {'identifier'}:
                    continue
                if k in r and r[k] == v:
                    del r[k]

        if metadata:
            r.update(self.metadata)
            r['uuid'] = self.uuid
        else:
            for k in self.metadata:
                if k in r:
                    del r[k]

        from superduper.components.datatype import DEFAULT_SERIALIZER

        s = None
        if schema:
            s = self.class_schema

        if self.__class__.__module__ == '__main__':
            return Document(
                {
                    '_object': DEFAULT_SERIALIZER.encode_data(
                        self.__class__, builds={}, blobs={}, files={}
                    ),
                    **r,
                },
                schema=s,
            )

        _path = f'{self.__class__.__module__}.{self.__class__.__name__}'
        if path:
            return Document({'_path': _path, **r}, schema=s)
        else:
            return Document(r, schema=s)

    @classmethod
    def _register_class(cls):
        """Register class in the class registry and set the full import path."""
        full_import_path = f"{cls.__module__}.{cls.__name__}"
        cls.full_import_path = full_import_path
        _CLASS_REGISTRY[full_import_path] = cls

    def unpack(self):
        """Unpack object."""
        return self

    @classmethod
    def build(cls, r):
        """Build object from an encoded data.

        :param r: Encoded data.
        """
        modified = {
            k: v
            for k, v in r.items()
            if k in inspect.signature(cls.__init__).parameters
        }
        return cls(**modified)

    def init(self, db=None):
        """Initialize object.

        :param db: Datalayer instance.
        """
        pass


def find_leaf_cls(full_import_path) -> t.Type[Leaf]:
    """Find leaf class by class full import path.

    :param full_import_path: Full import path of the class.
    """
    return _CLASS_REGISTRY[full_import_path]


class Address(Leaf):
    """Address is a base class for all address classes."""

    def __getattr__(self, name):
        try:
            super().__getattribute__(name)
        except AttributeError:
            return Attribute(
                identifier=f'{self.identifier}/{name}', parent=self, attribute=name
            )

    def __getitem__(self, item):
        return Index(identifier=f'{self.identifier}[{item}]', parent=self, index=item)

    def __call__(self, *args, **kwargs):
        return self.compile()(*args, **kwargs)

    def compile(self):
        """Compile the address."""
        raise NotImplementedError


class Import(Address):
    """
    Import is a class that imports a class from a module.

    :param import_path: The import path of the class.
    :param parent: The parent class.
    :param args: Positional arguments to pass to the class.
    :param kwargs: Keyword arguments to pass to the class.
    """

    import_path: str | None
    parent: dc.InitVar[t.Any | None] = None
    args: t.List | None = None
    kwargs: t.Dict | None = None

    def __post_init__(self, db: t.Optional['Datalayer'] = None, parent=None):
        super().__post_init__(db=db)

        assert parent is not None or self.import_path is not None
        if self.import_path is None:
            module = parent.__module__
            name = parent.__name__
            self.import_path = f'{module}.{name}'
        elif parent is None:
            module = '.'.join(self.import_path.split('.')[:-1])
            name = self.import_path.split('.')[-1]
            module = importlib.import_module(module)
            parent = getattr(module, name)

        if self.args is not None:
            assert self.kwargs is not None
            parent = parent.f(*self.args, **self.kwargs)
        self.parent = parent

    def compile(self):
        """Compile the import."""
        return self.parent


class ImportCall(Address):
    """
    ImportCall is a class that imports a function from a module and calls it.

    :param import_path: The import path of the function.
    :param args: Positional arguments to pass to the function.
    :param kwargs: Keyword arguments to pass to the function.
    """

    import_path: str
    args: t.Tuple = ()
    kwargs: t.Dict = dc.field(default_factory=dict)

    def __post_init__(self, db: t.Optional['Datalayer'] = None):
        super().__post_init__(db=db)
        name = self.import_path.split('.')[-1]
        module = '.'.join(self.import_path.split('.')[:-1])
        module = importlib.import_module(module)
        object = getattr(module, name)
        self.parent = object(*self.args, **self.kwargs)

    def compile(self):
        """Compile the import call."""
        return self.parent


class Attribute(Address):
    """
    An Attribute is a class that represents an attribute of a parent class.

    :param parent: The parent class.
    :param attribute: The attribute to get.
    """

    parent: Address
    attribute: str

    def compile(self):
        """Compile the attribute."""
        parent = self.parent.compile()
        return getattr(parent, self.attribute)


class Index(Address):
    """
    An Index is a class that represents an index of a parent class.

    :param parent: The parent class.
    :param index: The index to get.
    """

    parent: Leaf
    index: int

    def compile(self):
        """Compile the index."""
        parent = self.parent.compile()
        return parent[self.index]


def imported_value(f):
    """Wrap a import to be serialized as a fixed value.

    :param f: The function or class to wrap.
    """
    return Import(identifier=f.__name__, import_path=f'{f.__module__}.{f.__name__}')


def imported(f):
    """Wrap a function or class to be imported.

    :param f: The function or class to wrap.
    """
    if inspect.isclass(f):

        def wrapper(*args, **kwargs):
            return Import(
                identifier=f.__name__,
                import_path=f'{f.__module__}.{f.__name__}',
                args=args,
                kwargs=kwargs,
            )

        wrapper.f = f
        return wrapper
    else:

        def wrapper(*args, **kwargs):
            return ImportCall(
                identifier=f.__name__,
                import_path=f'{f.__module__}.{f.__name__}',
                args=args,
                kwargs=kwargs,
            )

        return wrapper
