import dataclasses as dc
import importlib
import inspect
import typing as t
from abc import ABC, abstractmethod, abstractproperty
from superduperdb.misc.serialization import asdict

_CLASS_REGISTRY = {}


def _import_item(cls, module, dict):
    module = importlib.import_module(module)
    cls = getattr(module, cls)
    try:
        return cls(**dict)
    except TypeError as e:
        if 'got an unexpected keyword argument' in str(e):
            if callable(cls):
                return cls(**{k: v for k, v in dict.items() if k in inspect.signature(cls).parameters})
            return cls(**{k: v for k, v in dict.items() if k in inspect.signature(cls.__init__).parameters})
        raise e


@dc.dataclass
class Leaf(ABC):
    """Base class for all leaf classes."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._register_class()

    def _deep_flat_encode(self, cache, blobs, files):
        from superduperdb.base.document import _deep_flat_encode
        r = dict(self.dict())
        return _deep_flat_encode(r, cache, blobs, files)

    def dict(self):
        from superduperdb import Document
        r = asdict(self)
        return Document({
            'module': self.__class__.__module__,
            'cls': self.__class__.__name__,
            'dict': r
        })

    @classmethod
    def handle_integration(cls, r):
        """Method to handle integration.

        :param r: Encoded data.
        """
        return r

    @classmethod
    def _register_class(cls):
        """Register class in the class registry and set the full import path."""
        full_import_path = f"{cls.__module__}.{cls.__name__}"
        cls.full_import_path = full_import_path
        _CLASS_REGISTRY[full_import_path] = cls

    @abstractproperty
    def unique_id(self):
        """Unique identifier for the object."""
        pass

    def unpack(self, db=None):
        """Unpack object.

        :param db: Datalayer instance.
        """
        return self

    @abstractmethod
    def encode(
        self,
        leaf_types_to_keep: t.Sequence = (),
    ):
        """Convert object to a saveable form.

        :param leaf_types_to_keep: Leaf types to keep.
        """
        pass

    @classmethod
    @abstractmethod
    def decode(cls, r, db=None):
        """Decode object from a saveable form.

        :param r: Encoded data.
        :param db: Datalayer instance.
        """
        pass

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
