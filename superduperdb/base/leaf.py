import dataclasses as dc
import importlib
import inspect
import typing as t
import uuid
from abc import ABC

from superduperdb.misc.serialization import asdict
from superduperdb.misc.special_dicts import SuperDuperFlatEncode

_CLASS_REGISTRY = {}

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.schema import Schema


def _import_item(cls, module, dict, db: t.Optional['Datalayer'] = None):
    module = importlib.import_module(module)
    cls = getattr(module, cls)
    try:
        return cls(**dict, db=db)
    except TypeError as e:
        if 'got an unexpected keyword argument' in str(e):
            if callable(cls) and not inspect.isclass(cls):
                return cls(
                    **{
                        k: v
                        for k, v in dict.items()
                        if k in inspect.signature(cls).parameters
                    },
                    db=db,
                )
            init_params = {
                k: v
                for k, v in dict.items()
                if k in inspect.signature(cls.__init__).parameters
            }
            post_init_params = {k: v for k, v in dict.items() if k in cls.set_post_init}
            instance = cls(**init_params, db=db)
            for k, v in post_init_params.items():
                setattr(instance, k, v)
            return instance
        raise e


@dc.dataclass
class Leaf(ABC):
    """Base class for all leaf classes.

    :param identifier: Identifier of the leaf.
    :param db: Datalayer instance.
    :param uuid: UUID of the leaf.
    """

    set_post_init: t.ClassVar[t.Sequence[str]] = ()

    identifier: str
    db: dc.InitVar[t.Optional['Datalayer']] = None
    uuid: str = dc.field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self, db):
        self.db: 'Datalayer' = db

    @property
    def _id(self):
        return f'{self.__class__.__name__.lower()}/{self.uuid}'

    def encode(self, schema: t.Optional['Schema'] = None, leaves_to_keep=()):
        """Encode itself.

        After encoding everything is a vanilla dictionary (JSON + bytes).

        :param schema: Schema instance.
        :param leaves_to_keep: Leaves to keep.
        """
        cache: t.Dict[str, dict] = {}
        blobs: t.Dict[str, bytes] = {}
        files: t.Dict[str, str] = {}

        self._deep_flat_encode(cache, blobs, files, leaves_to_keep, schema)
        return SuperDuperFlatEncode(
            {
                '_base': f'?{self._id}',
                '_leaves': cache,
                '_blobs': blobs,
                '_files': files,
            }
        )

    def set_variables(self, **kwargs) -> 'Leaf':
        """Set free variables of self.

        :param db: Datalayer instance.
        :param kwargs: Keyword arguments to pass to `_replace_variables`.
        """
        from superduperdb import Document
        from superduperdb.base.variables import Variable, _replace_variables

        r = self.dict().encode(leaves_to_keep=(Variable,))
        r = _replace_variables(r, **kwargs)
        return Document.decode(r).unpack()

    @property
    def variables(self) -> t.List[str]:
        """Get list of variables in the object."""
        from superduperdb.base.variables import Variable, _find_variables

        return list(set(_find_variables(self.encode(leaves_to_keep=Variable))))

    def _deep_flat_encode(
        self,
        cache,
        blobs,
        files,
        leaves_to_keep=(),
        schema: t.Optional['Schema'] = None,
    ):
        if isinstance(self, leaves_to_keep):
            cache[self._id] = self
            return f'?{self._id}'
        from superduperdb.base.document import _deep_flat_encode

        r = dict(self.dict())
        return _deep_flat_encode(
            r, cache, blobs, files, leaves_to_keep=leaves_to_keep, schema=schema
        )

    def dict(self):
        """Return dictionary representation of the object."""
        from superduperdb import Document

        r = asdict(self)

        path = (f'{self.__class__.__module__}.' f'{self.__class__.__name__}').replace(
            '.', '/'
        )
        return Document({'_path': path, **r})

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

    def init_from_blobs(self, blobs):
        """
        Initialize object from blobs.

        :param blobs: Blobs dictionary to init object.
        """

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
