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
        from superduper.base.datatype import INBUILT_DATATYPES
        from superduper.base.schema import Schema

        named_fields = cls._new_fields
        for f in named_fields:
            fields[f] = INBUILT_DATATYPES[named_fields[f]]

        out = Schema(fields=fields)
        return out

    @lazy_classproperty
    def table(cls):
        from superduper import Component
        from superduper.components.table import Table
        from superduper.misc.importing import isreallyinstance

        return Table(
            identifier=cls.__name__,
            fields=cls.class_schema.fields,
            path=cls.__module__ + '.' + cls.__name__,
            primary_id='uuid',
            is_component=isreallyinstance(cls, Component),
        )

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
        try:
            out = cls(**{k: v for k, v in r.items() if k != '_path'}, db=db)
            return out
        except TypeError as e:
            if 'got an unexpected keyword argument' in str(e):
                r['db'] = db
                signature_params = inspect.signature(cls.__init__).parameters
                init_params = {k: v for k, v in r.items() if k in signature_params}
                post_init_params = {
                    k: v for k, v in r.items() if k in cls.set_post_init
                }
                instance = cls(**init_params)
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

    @staticmethod
    def decode(r, db: t.Optional['Datalayer'] = None):
        """Decode a dictionary component into a `Component` instance.

        :param r: Object to be decoded.
        :param db: Datalayer instance.
        """
        from superduper.base.document import Document

        if '_path' in r:
            from superduper.misc.importing import import_object

            cls = import_object(r['_path'])

        r = Document.decode(r, schema=cls.class_schema, db=db)
        return cls.from_dict(r, db=None)

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

        r = self.dict()
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

        return {
            **r,
            KEY_BUILDS: context.builds,
            KEY_BLOBS: context.blobs,
            KEY_FILES: context.files,
        }

    # TODO needed?
    def set_variables(self, db: t.Union['Datalayer', None] = None, **kwargs) -> 'Base':
        """Set free variables of self.

        :param db: Datalayer instance.
        :param kwargs: Keyword arguments to pass to `_replace_variables`.
        """
        from superduper import Document
        from superduper.base.variables import _replace_variables

        r = self.encode()
        rr = _replace_variables(r, **kwargs)
        decoded = Document.decode(rr, schema=self.class_schema, db=db)
        return self.from_dict(decoded, db=db)

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

    def dict(self):
        """Return dictionary representation of the object."""
        from superduper import Document

        r = asdict(self)
        r['_path'] = self.__class__.__module__ + '.' + self.__class__.__name__
        return Document(r, schema=self.class_schema)

    @classmethod
    def build(cls, r):
        """Build object from an encoded data.

        :param r: Encoded data.
        """
        signature_params = inspect.signature(cls.__init__).parameters
        modified = {k: v for k, v in r.items() if k in signature_params}
        return cls(**modified)

    def setup(self, db=None):
        """Initialize object.

        :param db: Datalayer instance.
        """
        pass
