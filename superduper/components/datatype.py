import base64
import dataclasses as dc
import hashlib
import inspect
import json
import os
import pickle
import re
import typing as t
from abc import abstractmethod
from functools import cached_property
from importlib import import_module

import dill
import numpy

from superduper import CFG
from superduper.base.base import Base
from superduper.components.component import Component
from superduper.misc.utils import str_shape

Decode = t.Callable[[bytes], t.Any]
Encode = t.Callable[[t.Any], bytes]

# TODO do we need encodable?


@dc.dataclass
class FieldType:
    """Field type to represent the type of a field in a table.

    This is a wrapper around a database's native datatypes.

    :param identifier: The identifier of the datatype.
    """

    identifier: str

    def __repr__(self):
        return self.identifier


ID = FieldType(identifier='ID')


if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


def _convert_base64_to_bytes(str_: str) -> bytes:
    return base64.b64decode(str_)


def _convert_bytes_to_base64(bytes_: bytes) -> str:
    return base64.b64encode(bytes_).decode('utf-8')


# TODO - shouldn't this be handled by the datatype?
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


@dc.dataclass
class BaseDataType:
    """Base class for datatype."""

    dtype: t.ClassVar[str] = 'str'

    def __repr__(self):
        return f'{self.__class__.__name__}'

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
    def decode_data(self, item, builds, db):
        """Decode the item from bytes.

        :param item: The item to decode.
        :param builds: The build-cache dictionary.
        :param db: The datalayer.
        """


class ComponentType(BaseDataType):
    """Datatype for encoding leafs."""

    dtype: t.ClassVar[str] = 'str'
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
                f"{item.__class__.__name__}:{item.identifier}"
                if isinstance(item, Component)
                else item.identifier
            )
            builds[key] = item
            return '?' + key

        r = item.dict()
        # TODO why this clause?
        if r.schema:
            r = dict(
                r.schema.encode_data(
                    r, builds, blobs, files, leaves_to_keep=leaves_to_keep
                )
            )
        else:
            r = dict(r)

        identifier = r.pop('identifier')

        if '_schema' in r:
            del r['_schema']

        key = f"{item.__class__.__name__}:{identifier}"
        builds[key] = r
        return '?' + key

    def decode_data(self, item, builds, db):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        :param builds: The build-cache dictionary.
        :param db: The Datalayer.
        """
        if isinstance(item, str) and item.startswith('?'):
            key = item[1:]
            if isinstance(builds[key], dict):
                r = {'identifier': key.split(':')[-1], **builds[key]}
                builds[key] = _decode_leaf(r, builds, db=db)
            return builds[key]

        elif isinstance(item, str) and item.startswith('&'):
            _, component, _, uuid = item[2:].split(':')
            return db.load(component=component, uuid=uuid)

        elif isinstance(item, str):
            raise ValueError(f'Unknown reference type {item} for a leaf')

        out = _decode_leaf(item, builds, db=db)

        key = f"{out.__class__.__name__}:{out.identifier}"
        builds[key] = out

        return out


class LeafType(BaseDataType):
    """Datatype for encoding leafs."""

    dtype: t.ClassVar[str] = 'json'
    encodable: t.ClassVar[str] = 'leaf'

    def encode_data(self, item, builds, blobs, files, leaves_to_keep=()):
        """Encode the item as `bytes`.

        :param item: The item to encode.
        :param builds: The build-cache dictionary.
        :param blobs: The cache of blobs (bytes).
        :param files: The cache of files (paths).
        :param leaves_to_keep: The `Leaf` type(s) to keep.
        """
        r = item.dict()
        if r.schema:
            r = dict(
                r.schema.encode_data(
                    r, builds, blobs, files, leaves_to_keep=leaves_to_keep
                )
            )
        else:
            r = dict(r)

        if '_schema' in r:
            del r['_schema']

        return r

    def decode_data(self, item, builds, db):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        :param builds: The build-cache dictionary.
        :param db: The Datalayer.
        """
        out = _decode_leaf(item, builds, db=db)
        return out


def _decode_leaf(r, builds, db: t.Optional['Datalayer'] = None):
    assert '_path' in r
    cls = Base.get_cls_from_path(path=r['_path'])
    dict_ = {k: v for k, v in r.items() if k != '_path'}

    if inspect.isfunction(cls):
        out = cls(
            **{
                k: v for k, v in dict_.items() if k in inspect.signature(cls).parameters
            },
            db=db,
        )
    else:
        assert issubclass(cls, Base)
        dict_ = cls.class_schema.decode_data(dict_, builds=builds, db=db)
        out = cls.from_dict(dict_, db=db)

    if isinstance(out, Component):
        builds[f'{out.type_id}:{out.identifier}'] = out
    return out


class SDict(BaseDataType):
    """Datatype for encoding dictionaries which are supported as dict by databackend."""

    dtype: t.ClassVar[str] = 'dict'
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
                ComponentType().encode_data(
                    v, builds, blobs, files, leaves_to_keep=leaves_to_keep
                )
                if isinstance(v, Base)
                else v
            )
            for k, v in item.items()
        }

    def decode_data(self, item, builds, db):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        :param builds: The build-cache dictionary.
        :param db: The Datalayer.
        """
        return {k: ComponentType().decode_data(v, builds, db) for k, v in item.items()}


class SList(BaseDataType):
    """Datatype for encoding lists which are supported as list by databackend."""

    dtype: t.ClassVar[str] = 'json'
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
                ComponentType().encode_data(
                    r, builds, blobs, files, leaves_to_keep=leaves_to_keep
                )
                if isinstance(r, Base)
                else r
            )
            for r in item
        ]

    def decode_data(self, item, builds, db):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        :param builds: The builds.
        :param db: The Datalayer.
        """
        return [ComponentType().decode_data(r, builds, db) for r in item]


@dc.dataclass(kw_only=True)
class BaseVector(BaseDataType):
    """Base class for vector.

    :param shape: size of vector
    :param dtype: Datatype of array to encode.
    """

    shape: int
    dtype: str = 'float64'

    def __repr__(self):
        return f'{self.__class__.__name__}[{self.dtype}:{self.shape}]'

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
    def decode_data(self, item, builds, db):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        :param builds: The build cache.
        :param db: The Datalayer.
        """


@dc.dataclass(kw_only=True)
class NativeVector(BaseVector):
    """Datatype for encoding vectors which are supported as list by databackend.

    :param dtype: Datatype of encoded arrays.
    :param shape: Shape of array.
    """

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

    def decode_data(self, item, builds, db):
        """Decode the item from a list of floats.

        :param item: The item to decode.
        :param builds: The build cache.
        :param db: The Datalayer.
        """
        return numpy.array(item).astype(self.dtype)


@dc.dataclass(kw_only=True)
class Vector(BaseVector):
    """Vector meta-datatype for encoding vectors ready for search.

    :param dtype: Datatype of encoded arrays.
    :param shape: Shape of array.
    """

    @property
    def encodable(self):
        return self.datatype_impl.encodable

    @cached_property
    def datatype_impl(self):
        type_ = CFG.datatype_presets.vector
        if type_ is None:
            return NativeVector(shape=self.shape, dtype=self.dtype)

        module = '.'.join(type_.split('.')[:-1])
        cls = type_.split('.')[-1]
        datatype = getattr(import_module(module), cls)
        if inspect.isclass(datatype):
            datatype = datatype(dtype=self.dtype, shape=self.shape)
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

    def decode_data(self, item, builds, db):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        :param builds: The build cache.
        :param db: The Datalayer.
        """
        return self.datatype_impl.decode_data(item, builds, db)


class JSON(BaseDataType):
    """Datatype for encoding json-able items."""

    encodable: t.ClassVar[str] = 'native'
    dtype: t.ClassVar[str] = 'json'

    def encode_data(self, item, builds, blobs, files, leaves_to_keep=()):
        """Encode the item as a string.

        :param item: The item to encode.
        :param builds: The build-cache dictionary.
        :param blobs: The cache of blobs (bytes).
        :param files: The cache of files (paths).
        :param leaves_to_keep: The `Leaf` type(s) to keep.
        """
        if self.dtype == 'json':
            # necessary to enforce json-able content
            try:
                json.dumps(item)
            except Exception:
                raise TypeError(f'Item {item} is not json-able')
            return item
        return json.dumps(item)

    def decode_data(self, item, builds, db):
        """Decode the item from string form.

        :param item: The item to decode.
        :param builds: The build cache.
        :param db: The Datalayer.
        """
        if self.dtype == 'json':
            return item
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
        encoded = _convert_bytes_to_base64(encoded)
        return encoded

    def decode_data(self, item, builds, db):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        :param builds: The build cache.
        :param db: The Datalayer.
        """
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
        if isinstance(item, Blob):
            b = item
            b.init()
        else:
            b = Blob(
                bytes=self._encode_data(item),
                builder=self._decode_data,
            )
        blobs[b.identifier] = b.bytes
        return b.reference

    def decode_data(self, item, builds, db):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        :param builds: The build cache
        :param db: The datalayer.
        """
        if isinstance(item, Blob):
            return item
        assert isinstance(item, str)
        assert item.startswith('&')
        return Blob(
            identifier=item.split(':')[-1],
            builder=self._decode_data,
            db=db,
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

    def _decode_data(self, item):
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

    def decode_data(self, item, builds, db):
        """Decode the item placeholder.

        :param item: The file path to decode.
        :param builds: The build cache.
        :param db: Datalayer.
        """
        return FileItem(identifier=item.split(':')[-1], db=db)


@dc.dataclass(kw_only=True)
class Array(BaseDataType):
    """Encode/ decode a numpy array as bytes.

    :param dtype: numpy native datatype.
    :param shape: Shape of array.
    """

    encodable: t.ClassVar[str] = 'encodable'

    dtype: str = 'float64'
    shape: int | t.Tuple[int]

    def __repr__(self):
        return f'{self.__class__.__name__}[{self.dtype}:{str_shape(self.shape)}]'

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
        return _convert_bytes_to_base64(memoryview(item).tobytes())

    def decode_data(self, item, builds, db):
        """Decode the data.

        :param item: The data to decode.
        :param builds: The build cache
        :param db: The datalayer.
        """
        shape = self.shape
        if isinstance(shape, int):
            shape = (self.shape,)
        return numpy.frombuffer(
            _convert_base64_to_bytes(item), dtype=self.dtype
        ).reshape(shape)


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
    def create(data: t.Any) -> str:
        """Create a numpy array datatype.

        Used from creating an auto schema.

        :param data: The numpy array.
        """
        return f'array[{str(data.dtype)}:{str_shape(data.shape)}]'


class _DatatypeLookup:
    presets = {
        str(x).lower(): x
        for x in [
            JSON(),
            PickleEncoder(),
            DillEncoder(),
            Dill(),
            Pickle(),
            File(),
            LeafType(),
            ComponentType(),
            SDict(),
            SList(),
            FieldType('str'),
            FieldType('int'),
            FieldType('bytes'),
            FieldType('float'),
            FieldType('bool'),
            FieldType('ID'),
        ]
    }

    def __getitem__(self, item):
        try:
            return self.presets[item.lower()]
        except KeyError:

            import_match = re.match(r'^[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+$', item)

            if import_match:
                # only supports datatypes without arguments
                module, cls = item.rsplit('.', 1)
                return getattr(import_module(module), cls)

            vector_match = re.match(r'^vector\[([a-z0-9]+):([0-9]+)\]', item)

            if vector_match:
                dtype, shape = vector_match.groups()
                shape = int(shape)
                return Vector(dtype=dtype, shape=shape)

            array_match = re.match(r'^array\[([a-z0-9]+):(.*)\]$', item)

            if array_match:
                dtype, shape = array_match.groups()
                shape = tuple([int(x) for x in shape.split('x')])
                return Array(dtype=dtype, shape=shape)

            raise KeyError(f'Unknown datatype {item}')


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


@dc.dataclass
class Saveable:
    """A Saveable base class.

    :param identifier: Identifier of the object.
    :param db: The Datalayer.
    """

    identifier: str = ''
    db: 'Datalayer' = None

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


@dc.dataclass(kw_only=True)
class FileItem(Saveable):
    """Placeholder for a file.

    :param identifier: Identifier of the file.
    :param db: The Datalayer.
    :param path: Path to file.
    """

    path: str = ''

    def __post_init__(self):
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


@dc.dataclass(kw_only=True)
class Blob(Saveable):
    """Placeholder for a blob of bytes.

    :param bytes: The `bytes` blob.
    :param identifier: Identifier of the blob.
    :param builder: Function to rebuild object from bytes.
    :param db: The Datalayer.
    """

    bytes: bytearray | None = None
    identifier: str = ''
    builder: t.Callable

    def __post_init__(self):
        if not self.identifier:
            assert self.bytes is not None
            self.identifier = get_hash(self.bytes)

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


INBUILT_DATATYPES = _DatatypeLookup()

dill_serializer = Dill()
pickle_serializer = Pickle()
pickle_encoder = PickleEncoder()
file = File()

DEFAULT_SERIALIZER = dill_serializer
DEFAULT_ENCODER = pickle_encoder
