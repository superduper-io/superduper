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

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


RESERVED_INIT_PARAMS = frozenset(
    {
        "component",
        "uuid",
        "status",
        "details",
        "version",
    }
)


class _UniqueRegistry:
    def __init__(self, d):
        self.d = d

    def __contains__(self, item):
        if '.' in item:
            return item in self.d
        else:
            return item in [k.split('.')[-1] for k in self.d]

    def drop(self):
        for k in list(self.d.keys()):
            del self.d[k]

    def __getitem__(self, item):
        if '.' in item:
            return self.d[item]
        else:
            try:
                key = next(k for k in self.d if k.split('.')[-1] == item)
            except StopIteration:
                raise KeyError(item)
            return self.d[key]

    def __setitem__(self, key, value):
        matching = [x for x in self.d.keys() if x.split('.')[-1] == key.split('.')[-1]]
        if matching and matching[0] != key:
            raise ValueError(
                f"Class name \"{key.split('.')[-1]}\" already exists"
                f" as a `superduper` schema: as {matching[0]}."
            )
        elif matching:
            return
        self.d[key] = value

    def __repr__(self):
        return f'_UniqueRegistry({self.d})'


REGISTRY = _UniqueRegistry({})


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
            # Derived classes: kw_only=True
        else:
            # Base class: kw_only=False
            dataclass_params['kw_only'] = False

        cls = dc.dataclass(**dataclass_params, repr=False)(
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

    verbosity: t.ClassVar[int] = 0
    primary_id: t.ClassVar[str] = 'uuid'
    metadata_fields: t.ClassVar[t.Dict[str, str]] = {'_path': 'str'}

    def __init_subclass__(cls):
        full_path = f"{cls.__module__}.{cls.__name__}"
        REGISTRY[full_path] = cls

        sig = inspect.signature(cls.__init__)
        params = {name for name in sig.parameters if name != "self"}
        params.discard("db")

        bad = params & RESERVED_INIT_PARAMS
        if bad:
            raise TypeError(
                f"{cls.__module__}.{cls.__name__} uses reserved __init__ "
                f"parameter(s): {', '.join(sorted(bad))}. "
                "Rename these fields, "
                "so they are not part of the __init__ signature."
            )

        return super().__init_subclass__()

    @lazy_classproperty
    def _new_fields(cls):
        """Get the schema of the class."""
        from superduper.misc.schema import get_schema

        s = get_schema(cls)[0]
        return s

    @property
    def metadata(self):
        """Get metadata of the component."""
        return {k: getattr(self, k) for k in self.metadata_fields}

    @property
    def _path(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}"

    def set_variables(self, uuid_swaps: t.Dict | None = None, **kwargs) -> 'Base':
        """Set free variables of self.

        :param uuid_swaps: Dictionary of UUID swaps to apply.
        :param kwargs: Keyword arguments to pass to `_replace_variables`.
        """
        from superduper.base.variables import _replace_variables

        r = self.dict()
        rr = _replace_variables(r, uuid_swaps=uuid_swaps, **kwargs)
        out = self.from_dict(rr, db=self.db)
        return out

    @lazy_classproperty
    def class_schema(cls):
        fields = {}
        from superduper.base.datatype import INBUILT_DATATYPES
        from superduper.base.schema import Schema

        named_fields = cls._new_fields
        for f in named_fields:
            fields[f] = INBUILT_DATATYPES[named_fields[f]]

        out = Schema(fields=fields)
        return out

    @lazy_classproperty
    def table(cls):
        from superduper.components.table import Table

        if cls.__name__ == 'Base':
            return None
        return Table(
            identifier=cls.__name__,
            fields=cls.class_schema.fields,
            primary_id=cls.primary_id,
        )

    @staticmethod
    def get_cls_from_path(path):
        """Get class from a path.

        :param path: Import path to the class.
        """
        if path in REGISTRY:
            return REGISTRY[path]

        from superduper import logging

        logging.info(f'Importing: {path}')
        parts = path.split('.')
        cls = parts[-1]
        module = '.'.join(parts[:-1])
        import time

        start = time.time()
        module = importlib.import_module(module)
        logging.info(f'Imported {path} in {time.time() - start:.2f} seconds')
        out = getattr(module, cls)
        REGISTRY[path] = out
        return out

    @staticmethod
    def get_cls_from_blob(blob_ref, db):
        """Get class from a blob reference.

        :param blob_ref: Blob reference identifier.
        :param db: Datalayer instance.
        """
        from superduper.base.datatype import DEFAULT_SERIALIZER, Blob

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

    @lazy_classproperty
    def pydantic(cls):
        """Get the Pydantic model of the class."""
        from .schema import create_pydantic

        return create_pydantic(name=cls.__name__, schema=cls.class_schema)

    @lazy_classproperty
    def source_code(cls):
        """Get the source code of the class."""
        import inspect

        try:
            return inspect.getsource(cls)
        except OSError:
            from superduper import logging

            logging.warn(
                f"Could not get source code for {cls.__name__} from {cls.__module__} "
                "using inspect.getsource. "
                "Falling back to IPython history. "
                "This may not work in all environments."
            )
            from superduper.misc.utils import grab_source_code_ipython

            return grab_source_code_ipython(cls)

    @classmethod
    def from_dict(cls, r: t.Dict, db: t.Optional['Datalayer'] = None):
        if hasattr(cls, '_alternative_init'):
            signature_params = inspect.signature(cls._alternative_init).parameters
            return cls._alternative_init(
                **{k: v for k, v in r.items() if k in signature_params},
                db=db,
            )
        signature_params = inspect.signature(cls.__init__).parameters
        in_signature = {k: v for k, v in r.items() if k in signature_params}
        in_metadata = {
            k: v for k, v in r.items() if k in getattr(cls, 'metadata_fields', {})
        }
        assert set(in_signature) | set(in_metadata) == set(
            r
        ), f'Unexpected parameters in dict not in signature or metadata fields of {cls.__name__}: {set(r) - (set(in_signature) | set(in_metadata))}'
        if 'db' in signature_params:
            out = cls(**in_signature, db=db)
        else:
            out = cls(**in_signature)
        for k, v in r.items():
            if k in getattr(cls, 'metadata_fields', {}):
                try:
                    setattr(out, k, v)
                except AttributeError:
                    pass  # can't set property
        return out

    def postinit(self):
        """Post-initialization method."""
        pass

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

    @classmethod
    def decode(cls, r, db: t.Optional['Datalayer'] = None):
        """Decode a dictionary component into a `Component` instance.

        :param r: Object to be decoded.
        :param db: Datalayer instance.
        """
        from superduper.base.document import Document

        if 'component' in r and r['component'] in REGISTRY:
            cls = REGISTRY[r['component']]
        elif '_path' in r:
            cls = cls.get_cls_from_path(r['_path'])

        r = Document.decode(r, schema=cls.class_schema, db=db)
        return cls.from_dict(r, db=db)

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
            context = EncodeContext(name=self.__class__.__name__)

        for k, v in kwargs.items():
            setattr(context, k, v)

        r = None
        if context.keep_variables:
            r = self._original_parameters
            if r is not None:
                if not context.metadata:
                    for k in self.metadata_fields:
                        if k in r:
                            del r[k]
        if r is None:
            r = self.dict(metadata=context.metadata)
            r['component'] = self.__class__.__name__

        if not context.defaults:
            for k, v in list(r.items()):
                if not v:
                    del r[k]
            if 'details' in r:
                del r['details']
            if 'status' in r:
                del r['status']
            if 'version' in r:
                del r['version']

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
            _KEYS_TO_DROP = ('uuid', 'status', 'details', 'version')
            context.builds = _replace_uuids_with_keys(context.builds)
            for v in context.builds.values():
                for k in _KEYS_TO_DROP:
                    if k in v:
                        del v[k]

            for k in _KEYS_TO_DROP:
                if k in r:
                    del r[k]

            r = _replace_uuids_with_keys(r)

        return {
            **r,
            KEY_BUILDS: context.builds,
            KEY_BLOBS: context.blobs,
            KEY_FILES: context.files,
        }

    # # TODO needed?
    # def set_variables(self, db: t.Union['Datalayer', None] = None, **kwargs) -> 'Base':
    #     """Set free variables of self.

    #     :param db: Datalayer instance.
    #     :param kwargs: Keyword arguments to pass to `_replace_variables`.
    #     """
    #     from superduper import Document
    #     from superduper.base.variables import _replace_variables

    #     r = self.encode()
    #     rr = _replace_variables(r, **kwargs)
    #     decoded = Document.decode(rr, schema=self.class_schema, db=db)
    #     return self.from_dict(decoded, db=db)

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

    def dict(self, metadata: bool = True) -> t.Dict[str, t.Any]:
        """Return dictionary representation of the object.

        :param metadata: Whether to include metadata in the dictionary.
        """
        from superduper import Document

        r = asdict(self)
        if metadata:
            metadata = getattr(self, 'metadata', {})
            for k, v in metadata.items():
                r[k] = v
        else:
            for k in list(r.keys()):
                if k in getattr(self, 'metadata_fields', {}):
                    if k in r:
                        del r[k]

        return Document(r, schema=self.class_schema)

    @classmethod
    def build(cls, r):
        """Build object from an encoded data.

        :param r: Encoded data.
        """
        signature_params = inspect.signature(cls.__init__).parameters
        modified = {k: v for k, v in r.items() if k in signature_params}
        return cls(**modified)

    def setup(self):
        """Initialize object."""
        pass
