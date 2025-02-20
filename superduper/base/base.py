import dataclasses as dc
import importlib
import inspect
import typing as t

from superduper.base.constant import KEY_BLOBS, KEY_BUILDS, KEY_FILES
from superduper.base.encoding import EncodeContext
from superduper.misc.annotations import (
    extract_parameters,
    lazy_classproperty,
    replace_parameters,
)
from superduper.misc.serialization import asdict

_CLASS_REGISTRY = {}

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class BaseMeta(type):
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

        is_base = (
            namespace.get('__module__', '') == 'superduper.base.base' and name == 'Base'
        )
        is_component = (
            namespace.get('__module__', '') == 'superduper.components.component'
            and name == 'Component'
        )
        is_base = is_base or is_component

        dataclass_params = namespace.get('_dataclass_params', {}).copy()
        if bases and any(dc.is_dataclass(b) for b in bases) and not is_base:
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


class Base(metaclass=BaseMeta):
    """Base class for all superduper classes."""

    set_post_init: t.ClassVar[t.Sequence[str]] = ()

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

    @property
    def leaves(self):
        """Get all leaves in the object."""
        return {
            f.name: getattr(self, f.name)
            for f in dc.fields(self)
            if isinstance(getattr(self, f.name), Base)
        }

    @classmethod
    def cls_encode(cls, item: 'Base', builds, blobs, files, leaves_to_keep=()):
        """Encode a dictionary component into a `Component` instance.

        :param r: Object to be encoded.
        """
        return item.encode(
            builds=builds, blobs=blobs, files=files, leaves_to_keep=leaves_to_keep
        )

    def encode(
        self,
        context: t.Optional['EncodeContext'] = None,
        **kwargs,
    ):
        """Encode itself.

        After encoding everything is a vanilla dictionary (JSON + bytes).

        :param context: Encoding context.
        :param kwargs: Additional encoding parameters.
        """
        if context is None:
            context = EncodeContext()

        for k, v in kwargs.items():
            setattr(context, k, v)

        r = self.dict(metadata=context.metadata, defaults=context.defaults)
        r = self.class_schema.encode_data(r, context=context)

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

        lookup = {v['uuid']: k for k, v in context.builds.items() if 'uuid' in v}

        context.builds = _replace_loads_with_references(context.builds, lookup)

        def _replace_uuids_with_keys(record):
            import json

            dump = json.dumps(record)
            for k, v in lookup.items():
                dump = dump.replace(k, f'?({v}.uuid)')
            return json.loads(dump)

        if not context.metadata:
            context.builds = _replace_uuids_with_keys(context.builds)
            for v in context.builds.values():
                if 'uuid' in v:
                    del v['uuid']

            if 'uuid' in r:
                del r['uuid']
            r = _replace_uuids_with_keys(r)

        # TODO deprecate this wrapper (not needed)
        return {
            **r,
            KEY_BUILDS: context.builds,
            KEY_BLOBS: context.blobs,
            KEY_FILES: context.files,
        }

    def set_variables(self, db: t.Union['Datalayer', None] = None, **kwargs) -> 'Base':
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

    def dict(
        self,
        metadata: bool = True,
        defaults: bool = True,
        path: bool = True,
    ):
        """Return dictionary representation of the object.

        :param metadata: Include metadata.
        :param defaults: Include default values.
        :param path: Include path.
        """
        from superduper import Document

        r = asdict(self)

        if not defaults:
            for k, v in self.defaults.items():
                if k in r and r[k] == v:
                    del r[k]

        if metadata:
            r.update(self.metadata)
        else:
            for k in self.metadata:
                if k in r:
                    del r[k]

        if path:
            if self.__class__.__module__ == '__main__':
                raise ValueError('Module name cannot be __main__')
            _path = f'{self.__class__.__module__}.{self.__class__.__name__}'
            if path:
                return Document({'_path': _path, **r}, schema=self.class_schema)
        else:
            return Document(r, schema=self.class_schema)

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


def find_leaf_cls(full_import_path) -> t.Type[Base]:
    """Find leaf class by class full import path.

    :param full_import_path: Full import path of the class.
    """
    return _CLASS_REGISTRY[full_import_path]
