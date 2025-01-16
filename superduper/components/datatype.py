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
    def encode_data(self, item):
        """Decode the item as `bytes`.

        :param item: The item to decode.
        """

    @abstractmethod
    def decode_data(self, item):
        """Decode the item from bytes.

        :param item: The item to decode.
        """


class BaseVector(BaseDataType):
    """Base class for vector.

    :param shape: size of vector

    :param dtype: Datatype of array to encode.
    """

    shape: int
    dtype: str = 'float64'

    @abstractmethod
    def encode_data(self, item):
        """Encode the item as `bytes`.

        :param item: The item to encode.
        """

    @abstractmethod
    def decode_data(self, item):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        """


class NativeVector(BaseVector):
    """Datatype for encoding vectors which are supported as list by databackend."""

    encodable: t.ClassVar[str] = 'native'
    dtype: str = 'float'

    def encode_data(self, item):
        """Encode the item as a list of floats.

        :param item: The item to encode.
        """
        if isinstance(item, numpy.ndarray):
            item = item.tolist()
        return item

    def decode_data(self, item):
        """Decode the item from a list of floats.

        :param item: The item to decode.
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

    def encode_data(self, item):
        """Encode the item as `bytes`.

        :param item: The item to encode.
        """
        return self.datatype_impl.encode_data(item=item)

    def decode_data(self, item):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        """
        return self.datatype_impl.decode_data(item=item)


class JSON(BaseDataType):
    """Datatype for encoding vectors which are supported natively by databackend.

    :param dtype: Datatype of encoded arrays.
    """

    encodable: t.ClassVar[str] = 'native'
    dtype: str = 'str'

    def __post_init__(self, db):
        return super().__post_init__(db)

    def encode_data(self, item):
        """Encode the item as a string.

        :param item: The dictionary or list (json-able object) to encode.
        """
        return json.dumps(item)

    def decode_data(self, item):
        """Decode the item from string form.

        :param item: The item to decode.
        """
        return json.loads(item)


class _Encodable:
    encodable: t.ClassVar[str] = 'encodable'

    def encode_data(self, item):
        """Encode the item as `bytes`.

        :param item: The item to encode.
        """
        return self._encode_data(item)


class _Artifact:
    encodable: t.ClassVar[str] = 'artifact'

    def encode_data(self, item):
        """Encode the item as `bytes`.

        :param item: The item to encode.
        """
        return Blob(bytes=self._encode_data(item))


class _PickleMixin:
    def _encode_data(self, item):
        return pickle.dumps(item)

    def decode_data(self, item):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        """
        return pickle.loads(item)


class Pickle(_Artifact, _PickleMixin, BaseDataType):
    """Serializer with pickle."""


class PickleEncoder(_Encodable, _PickleMixin, BaseDataType):
    """Pickle encoder."""


class _DillMixin:
    def _encode_data(self, item):
        return dill.dumps(item, recurse=True)

    def decode_data(self, item):
        """Decode the item from `bytes`.

        :param item: The item to decode.
        """
        return dill.loads(item)


class Dill(_Artifact, _DillMixin, BaseDataType):
    """Serializer with dill.

    This is also the default serializer.
    >>> from superduper.components.datatype import DEFAULT_SERIALIZER
    """


class DillEncoder(_Encodable, _DillMixin, BaseDataType):
    """Encoder with dill."""

    ...


class File(BaseDataType):
    """Type for encoding files on disk."""

    encodable: t.ClassVar[str] = 'file'

    def encode_data(self, item):
        """Encode the item as a file path.

        :param item: The file path to encode.
        """
        assert os.path.exists(item)
        return FileItem(path=item)

    def decode_data(self, item):
        """Decode the item placeholder.

        :param item: The file path to decode.
        """
        return item


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

    def __post_init__(self, db=None):
        if not self.identifier:
            self.identifier = get_hash(self.path)
        return super().__post_init__(db)

    def init(self):
        """Initialize the file to local disk."""
        if self.path:
            return
        self.path = self.db.artifact_store.get_file(self.identifier)

    def unpack(self):
        """Get the path out of the object."""
        self.init()
        return self.path

    # TODO - return this as self.dict()?
    @property
    def reference(self):
        return f'&:file:{self.identifier}'


class Blob(Saveable):
    """Placeholder for a blob of bytes.

    :param bytes: Bytes blob.
    """

    bytes: bytearray | None = None
    identifier: str = ''

    def __post_init__(self, db=None):
        if not self.identifier:
            assert self.bytes is not None
            self.identifier = get_hash(self.bytes)
        return super().__post_init__(db)

    # TODO why do some of these methods have `init(self, db=None)`?
    def init(self):
        """Initialize the blob."""
        if self.bytes:
            return
        self.bytes = self.db.artifact_store.get_bytes(self.identifier)

    def unpack(self):
        """Get the bytes out of the blob."""
        self.init()
        return self.bytes

    @property
    def reference(self):
        return f'&:blob:{self.identifier}'


json_encoder = JSON('json')
pickle_encoder = PickleEncoder('pickle_encoder')
pickle_serializer = Pickle('pickle_serializer')
dill_encoder = DillEncoder('dill_encoder')
dill_serializer = Dill('dill_serializer')
file = File('file')


INBUILT_DATATYPES.update(
    {
        dt.identifier: dt
        for dt in [
            json_encoder,
            pickle_encoder,
            pickle_serializer,
            dill_encoder,
            dill_serializer,
            file,
        ]
    }
)

DEFAULT_ENCODER = INBUILT_DATATYPES['PickleEncoder']
DEFAULT_SERIALIZER = INBUILT_DATATYPES['Dill']
INBUILT_DATATYPES['default'] = DEFAULT_SERIALIZER
INBUILT_DATATYPES['Blob'] = INBUILT_DATATYPES['Pickle']


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

    def encode_data(self, item):
        """Encode the data.

        :param item: The data to encode.
        """
        if item.dtype != self.dtype:
            raise TypeError(f'dtype was {item.dtype}, expected {self.dtype}')
        return memoryview(item).tobytes()

    def decode_data(self, item):
        """Decode the data.

        :param item: The data to decode.
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
    def create(data: t.Any) -> Array:
        """Create a numpy array datatype.

        Used from creating an auto schema.

        :param data: The numpy array.
        """
        return Array(dtype=str(data.dtype), shape=list(data.shape))
