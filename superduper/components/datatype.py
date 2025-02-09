import base64
import hashlib
import inspect
import json
import os
import pickle
import typing as t
from abc import abstractmethod
from functools import cached_property
from importlib import import_module

import dill
import numpy

from superduper import CFG
from superduper.base.leaf import Leaf
from superduper.components.component import Component, ComponentMeta
from superduper.misc.utils import str_shape

Decode = t.Callable[[bytes], t.Any]
Encode = t.Callable[[t.Any], bytes]

INBUILT_DATATYPES = {}


if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


def _convert_base64_to_bytes(str_: str) -> bytes:
    return base64.b64decode(str_)


def _convert_bytes_to_base64(bytes_: bytes) -> str:
    return base64.b64encode(bytes_).decode('utf-8')


class DataTypeFactory:
    """Abstract class for creating a DataType # noqa."""

    @staticmethod
    @abstractmethod
    def check(data: t.Any) -> bool:
        """Check if the data can be encoded by the DataType.

        If the data can be encoded, return True, otherwise False

        :param data: The data to check
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create(data: t.Any) -> "BaseDataType":
        """Create a DataType for the data.

        :param data: The data to create the DataType for
        """
        raise NotImplementedError


class DataTypeMeta(ComponentMeta):
    """Metaclass for the `Model` class and descendants # noqa."""

    def __new__(mcls, name, bases, dct):
        """Create a new class with merged docstrings # noqa."""
        cls = super().__new__(mcls, name, bases, dct)
        try:
            instance = cls(cls.__name__)
            INBUILT_DATATYPES[cls.__name__] = instance
        except TypeError:
            pass
        return cls


class BaseDataType(Component, metaclass=DataTypeMeta):
    """Base class for datatype."""

    type_id: t.ClassVar[str] = 'datatype'
    cache: bool = True

    @abstractmethod
    def encode_data(self, item, builds, blobs, files, leaves_to_keep=()):
        """Decode the item as `bytes`.

        :param item: The item to decode.
        :param builds: The build-cache dictionary.
        :param blobs: The cache of blobs (bytes).
        :param files: The cache of files (paths).
        :param leaves_to_keep: The `Leaf` type(s) to keep.
        """

    @abstractmethod
    def decode_data(self, item, *args, **kwargs):
        """Decode the item from bytes.

        :param item: The item to decode.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """


class LeafType(BaseDataType):
    """Datatype for encoding leafs."""

    encodable: t.ClassVar[str] = 'leaf'

    def encode_data(self, item, builds, blobs, files, leaves_to_keep=()):
        """Encode the item as `bytes`.

        :param item: The item to encode.
        :param builds: The build-cache dictionary.
        :param blobs: The cache of blobs (bytes).
        :param files: The cache of files (paths).
        :param leaves_to_keep: The `Leaf` type(s) to keep.
        """
        if isinstance(item, leaves_to_keep):
            key = (
                f"{item.type_id}:{item.identifier}"
                if hasattr(item, 'type_id')
                else item.identifier
            )
            builds[key] = item
            return '?' + key

        r = item.dict()
        if r.schema:
            r = dict(
                r.schema.encode_data(
                    r, builds, blobs, files, leaves_to_keep=leaves_to_keep
                )
            )

        identifier = r.pop('identifier')

        if '_schema' in r:
            del r['_schema']

        key = (
            f"{item.type_id}:{identifier}"
            if isinstance(item, Component)
            else identifier
        )
        builds[key] = r
        return '?' + key

    def decode_data(self, item, builds: t.Dict, *args, **kwargs):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        :param builds: The build-cache dictionary.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        from superduper.base.datalayer import Datalayer

        if isinstance(item, str) and item.startswith('?'):
            key = item[1:]
            if isinstance(builds[key], dict):
                r = {'identifier': key.split(':')[-1], **builds[key]}
                builds[key] = _decode_leaf(r, builds, *args, db=self.db, **kwargs)  # type: ignore[arg-type]
            return builds[key]

        elif isinstance(item, str) and item.startswith('&'):
            uuid = item[1:].split(':')[-1]
            return self.db.load(uuid=uuid)

        elif isinstance(item, str):
            raise ValueError(f'Unknown reference type {item} for a leaf')

        assert isinstance(self.db, Datalayer)
        out = _decode_leaf(item, builds, *args, db=self.db, **kwargs)

        if isinstance(out, Component):
            key = f"{out.type_id}:{out.identifier}"
        else:
            key = out.identifier

        builds[key] = out

        return out


def _decode_leaf(r, builds, db: t.Optional['Datalayer'] = None):
    if '_path' in r:
        cls = Leaf.get_cls_from_path(path=r['_path'])
    else:
        assert '_object' in r, 'Require _path or _blob in object'
        cls = Leaf.get_cls_from_blob(path=r['_object'])

    dict_ = {k: v for k, v in r.items() if k not in {'_object', '_path'}}

    if inspect.isfunction(cls):
        out = cls(
            **{
                k: v for k, v in dict_.items() if k in inspect.signature(cls).parameters
            },
            db=db,
        )
    else:
        assert issubclass(cls, Leaf)
        schema = cls.build_class_schema(db=db)
        dict_ = schema.decode_data(dict_, builds=builds)
        out = cls.from_dict(dict_, db=db)

    if isinstance(out, Leaf):
        builds[out.identifier] = out
    else:
        assert isinstance(out, Component)
        builds[f'{out.type_id}:{out.identifier}'] = out
    return out


class SDict(BaseDataType):
    """Datatype for encoding dictionaries which are supported as dict by databackend."""

    encodable: t.ClassVar[str] = 'native'

    def encode_data(self, item, builds, blobs, files, leaves_to_keep=()):
        """Encode the item as `bytes`.

        :param item: The item to encode.
        :param builds: The build-cache dictionary.
        :param blobs: The cache of blobs (bytes).
        :param files: The cache of files (paths).
        :param leaves_to_keep: The `Leaf` type(s) to keep.
        """
        assert isinstance(item, dict)
        return {
            k: (
                LeafType('leaf').encode_data(
                    v, builds, blobs, files, leaves_to_keep=leaves_to_keep
                )
                if isinstance(v, Leaf)
                else v
            )
            for k, v in item.items()
        }

    def decode_data(self, item, builds, *args, **kwargs):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        :param builds: The build-cache dictionary.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        leaf_type = LeafType('leaf', db=self.db)
        return {
            k: leaf_type.decode_data(v, builds, *args, **kwargs)
            for k, v in item.items()
        }


class SList(BaseDataType):
    """Datatype for encoding lists which are supported as list by databackend."""

    encodable: t.ClassVar[str] = 'native'

    def encode_data(self, item, builds, blobs, files, leaves_to_keep=()):
        """Encode the item as `bytes`.

        :param item: The item to encode.
        :param builds: The build-cache dictionary.
        :param blobs: The cache of blobs (bytes).
        :param files: The cache of files (paths).
        :param leaves_to_keep: The `Leaf` type(s) to keep.
        """
        assert isinstance(item, list)
        return [
            (
                LeafType('leaf').encode_data(
                    r, builds, blobs, files, leaves_to_keep=leaves_to_keep
                )
                if isinstance(r, Leaf)
                else r
            )
            for r in item
        ]

    def decode_data(self, item, builds, *args, **kwargs):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        :param builds: The builds.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        leaf_type = LeafType('leaf', db=self.db)
        return [leaf_type.decode_data(r, builds, *args, **kwargs) for r in item]


class BaseVector(BaseDataType):
    """Base class for vector.

    :param shape: size of vector

    :param dtype: Datatype of array to encode.
    """

    shape: int
    dtype: str = 'float64'

    @abstractmethod
    def encode_data(self, item, builds, blobs, files, leaves_to_keep=()):
        """Encode the item as `bytes`.

        :param item: The item to encode.
        :param builds: The build-cache dictionary.
        :param blobs: The cache of blobs (bytes).
        :param files: The cache of files (paths).
        :param leaves_to_keep: The `Leaf` type(s) to keep.
        """

    @abstractmethod
    def decode_data(self, item, *args, **kwargs):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """


class NativeVector(BaseVector):
    """Datatype for encoding vectors which are supported as list by databackend."""

    encodable: t.ClassVar[str] = 'native'
    dtype: str = 'float'

    def encode_data(self, item, builds, blobs, files, leaves_to_keep=()):
        """Encode the item as a list of floats.

        :param item: The item to encode.
        :param builds: The build-cache dictionary.
        :param blobs: The cache of blobs (bytes).
        :param files: The cache of files (paths).
        :param leaves_to_keep: The `Leaf` type(s) to keep.
        """
        if isinstance(item, numpy.ndarray):
            item = item.tolist()
        return item

    def decode_data(self, item, *args, **kwargs):
        """Decode the item from a list of floats.

        :param item: The item to decode.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        return numpy.array(item).astype(self.dtype)


class Vector(BaseVector):
    """Vector meta-datatype for encoding vectors ready for search.

    :param dtype: Datatype of encoded arrays.
    """

    identifier: str = ''

    def __post_init__(self, db):
        self.identifier = f'vector[{self.shape[0]}]'
        super().__post_init__(db)

    @property
    def encodable(self):
        return self.datatype_impl.encodable

    @cached_property
    def datatype_impl(self):
        if isinstance(CFG.datatype_presets.vector, str):
            type_: str = CFG.datatype_presets.vector
        else:
            type_: str = self.db.databackend.datatype_presets['vector']

        module = '.'.join(type_.split('.')[:-1])
        cls = type_.split('.')[-1]
        datatype = getattr(import_module(module), cls)
        if inspect.isclass(datatype):
            datatype = datatype('tmp', dtype=self.dtype, shape=self.shape)
        return datatype

    def encode_data(self, item, builds, blobs, files, leaves_to_keep=()):
        """Encode the item as `bytes`.

        :param item: The item to encode.
        :param builds: The build-cache dictionary.
        :param blobs: The cache of blobs (bytes).
        :param files: The cache of files (paths).
        :param leaves_to_keep: The `Leaf` type(s) to keep.
        """
        return self.datatype_impl.encode_data(
            item, builds, blobs, files, leaves_to_keep=leaves_to_keep
        )

    def decode_data(self, item, *args, **kwargs):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        return self.datatype_impl.decode_data(item=item)


class JSON(BaseDataType):
    """Datatype for encoding vectors which are supported natively by databackend.

    :param dtype: Datatype of encoded arrays.
    """

    encodable: t.ClassVar[str] = 'native'
    dtype: str = 'str'

    def encode_data(self, item, builds, blobs, files, leaves_to_keep=()):
        """Encode the item as a string.

        :param item: The item to encode.
        :param builds: The build-cache dictionary.
        :param blobs: The cache of blobs (bytes).
        :param files: The cache of files (paths).
        :param leaves_to_keep: The `Leaf` type(s) to keep.
        """
        return json.dumps(item)

    def decode_data(self, item, *args, **kwargs):
        """Decode the item from string form.

        :param item: The item to decode.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        return json.loads(item)


class _Encodable:
    encodable: t.ClassVar[str] = 'encodable'

    def encode_data(self, item, builds, blobs, files, leaves_to_keep=()):
        """Encode the item as `bytes`.

        :param item: The item to encode.
        :param builds: The build-cache dictionary.
        :param blobs: The cache of blobs (bytes).
        :param files: The cache of files (paths).
        :param leaves_to_keep: The `Leaf` type(s) to keep.
        """
        encoded = self._encode_data(item)
        if self.db.databackend.bytes_encoding == 'base64':
            encoded = _convert_bytes_to_base64(encoded)
        return encoded

    def decode_data(self, item, *args, **kwargs):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        if self.db.databackend.bytes_encoding == 'base64':
            item = _convert_base64_to_bytes(item)
        return self._decode_data(item)


class _Artifact:
    encodable: t.ClassVar[str] = 'artifact'

    def encode_data(self, item, builds, blobs, files, leaves_to_keep=()):
        """Encode the item as `bytes`.

        :param item: The item to encode.
        :param builds: The build-cache dictionary.
        :param blobs: The cache of blobs (bytes).
        :param files: The cache of files (paths).
        :param leaves_to_keep: The `Leaf` type(s) to keep.
        """
        b = Blob(
            bytes=self._encode_data(item),
            builder=self._decode_data,
            db=self.db,
        )
        blobs[b.identifier] = b.bytes
        return b.reference

    def decode_data(self, item, *args, **kwargs):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        assert isinstance(item, str)
        assert item.startswith('&')
        return Blob(
            identifier=item.split(':')[-1],
            builder=self._decode_data,
            db=self.db,
        )


class _PickleMixin:
    def _encode_data(self, item):
        return pickle.dumps(item)

    def _decode_data(self, item):
        return pickle.loads(item)


class Pickle(_Artifact, _PickleMixin, BaseDataType):
    """Serializer with pickle."""


class PickleEncoder(_Encodable, _PickleMixin, BaseDataType):
    """Pickle encoder."""


class _DillMixin:
    def _encode_data(self, item):
        return dill.dumps(item, recurse=True)

    def _decode_data(self, item, *args, **kwargs):
        return dill.loads(item)


class Dill(_Artifact, _DillMixin, BaseDataType):
    """Serializer with dill.

    This is also the default serializer.
    >>> from superduper.components.datatype import DEFAULT_SERIALIZER
    """


class DillEncoder(_Encodable, _DillMixin, BaseDataType):
    """Encoder with dill."""


class File(BaseDataType):
    """Type for encoding files on disk."""

    encodable: t.ClassVar[str] = 'file'

    def encode_data(self, item, builds, blobs, files, leaves_to_keep=()):
        """Encode the item as a file path.

        :param item: The item to encode.
        :param builds: The build-cache dictionary.
        :param blobs: The cache of blobs (bytes).
        :param files: The cache of files (paths).
        :param leaves_to_keep: The `Leaf` type(s) to keep.
        """
        assert os.path.exists(item)
        file = FileItem(path=item)
        files[file.identifier] = file.path
        return file.reference

    def decode_data(self, item, *args, **kwargs):
        """Decode the item placeholder.

        :param item: The file path to decode.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        return FileItem(identifier=item.split(':')[-1], db=self.db)


def get_hash(data):
    """Get the hash of the given data.

    :param data: Data to hash.
    """
    if isinstance(data, str):
        bytes_ = data.encode()
    elif isinstance(data, bytes):
        bytes_ = data
    else:
        bytes_ = str(id(data)).encode()
    return hashlib.sha1(bytes_).hexdigest()


class Saveable(Leaf):
    """A Saveable base class."""

    identifier: str = ''

    @property
    @abstractmethod
    def reference(self):
        pass

    @abstractmethod
    def init(self):
        """Initialize the object."""
        pass

    @abstractmethod
    def unpack(self):
        """Unpack the object to its original form."""
        pass


class FileItem(Saveable):
    """Placeholder for a file.

    :param path: Path to file.
    """

    path: str = ''

    def postinit(self):
        """Post init."""
        if not self.identifier:
            self.identifier = get_hash(self.path)

    def init(self):
        """Initialize the file to local disk."""
        if self.path:
            return
        self.path = self.db.artifact_store.get_file(self.identifier)

    def unpack(self):
        """Get the path out of the object."""
        self.init()
        return self.path

    @property
    def reference(self):
        return f'&:file:{self.identifier}'


class Blob(Saveable):
    """Placeholder for a blob of bytes.

    :param bytes: Bytes blob.
    :param builder: Function to rebuild object from bytes.
    """

    bytes: bytearray | None = None
    identifier: str = ''
    builder: t.Callable

    def __post_init__(self, db=None):
        if not self.identifier:
            assert self.bytes is not None
            self.identifier = get_hash(self.bytes)
        return super().__post_init__(db)

    def init(self):
        """Initialize the blob."""
        if self.bytes:
            return
        self.bytes = self.db.artifact_store.get_bytes(self.identifier)

    def unpack(self):
        """Get the bytes out of the blob."""
        if self.bytes is None:
            self.init()
        return self.builder(self.bytes)

    @property
    def reference(self):
        return f'&:blob:{self.identifier}'


class Array(BaseDataType):
    """Encode/ decode a numpy array as bytes.

    :param dtype: numpy native datatype.
    :param shape: Shape of array.
    """

    dtype: str = 'float64'
    shape: int | t.Tuple[int]
    identifier: str = ''

    def __post_init__(self, db):
        self.encodable = 'encodable'
        if not self.identifier:
            dtype = str(self.dtype)
            self.identifier = f'numpy-{dtype}[{str_shape(self.shape)}]'
        return super().__post_init__(db)

    def encode_data(self, item, builds, blobs, files, leaves_to_keep=()):
        """Encode the data.

        :param item: The item to encode.
        :param builds: The build-cache dictionary.
        :param blobs: The cache of blobs (bytes).
        :param files: The cache of files (paths).
        :param leaves_to_keep: The `Leaf` type(s) to keep.
        """
        if item.dtype != self.dtype:
            raise TypeError(f'dtype was {item.dtype}, expected {self.dtype}')
        return memoryview(item).tobytes()

    def decode_data(self, item, *args, **kwargs):
        """Decode the data.

        :param item: The data to decode.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        shape = self.shape
        if isinstance(shape, int):
            shape = (self.shape,)
        return numpy.frombuffer(item, dtype=self.dtype).reshape(shape)


class NumpyDataTypeFactory(DataTypeFactory):
    """A factory for numpy arrays # noqa."""

    @staticmethod
    def check(data: t.Any) -> bool:
        """Check if the data is a numpy array.

        Used for creating an auto schema.

        :param data: The data to check.
        """
        return isinstance(data, numpy.ndarray)

    @staticmethod
    def create(data: t.Any, db: 'Datalayer') -> Array:
        """Create a numpy array datatype.

        Used from creating an auto schema.

        :param data: The numpy array.
        :param db: The datalayer.
        """
        return Array(dtype=str(data.dtype), shape=list(data.shape))


INBUILT_DATATYPES = {
    'json': JSON,
    'encodable': PickleEncoder,
    'component': LeafType,
    'default': Dill,
    'blob': Dill,
    'leaf': LeafType,
    'file': File,
    'sdict': SDict,
    'slist': SList,
}

dill_serializer = Dill('dill_serializer')
pickle_serializer = Pickle('pickle_serializer')
pickle_encoder = PickleEncoder('pickle_encoder')
file = File('file')

DEFAULT_SERIALIZER = dill_serializer
DEFAULT_ENCODER = pickle_encoder
