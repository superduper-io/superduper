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
from superduper.components.component import Component

Decode = t.Callable[[bytes], t.Any]
Encode = t.Callable[[t.Any], bytes]


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


class BaseDataType(Component):
    """Base class for datatype."""

    type_id: t.ClassVar[str] = 'datatype'
    cache: bool = True

    @abstractmethod
    def encode_data(self, item):
        """Decode the item as `bytes`.

        :param item: The item to decode.
        :param info: The optional information dictionary.
        """

    @abstractmethod
    def decode_data(self, item):
        """Decode the item from bytes.

        :param item: The item to decode.
        :param info: The optional information dictionary.
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
        pass

    @abstractmethod
    def decode_data(self, item):
        pass


class NativeVector(BaseVector):
    """Datatype for encoding vectors which are supported as list by databackend."""

    encodable: t.ClassVar[str] = 'native'
    dtype: str = 'float'

    def encode_data(self, item):
        if isinstance(item, numpy.ndarray):
            item = item.tolist()
        return item

    def decode_data(self, item):
        # TODO:
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
        return self.datatype_impl.encode_data(item=item)

    def decode_data(self, item):
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
        return json.dumps(item)

    def decode_data(self, item):
        return json.loads(item)


class _Encodable:
    encodable: t.ClassVar[str] = 'encodable'

    def encode_data(self, item):
        return self._encode_data(item)


class _Artifact:
    encodable: t.ClassVar[str] = 'artifact'

    def encode_data(self, item):
        return Blob(bytes=self._encode_data(item))


class _PickleMixin:
    def _encode_data(self, item):
        return pickle.dumps(item)

    def decode_data(self, item):
        return pickle.loads(item)


class PickleSerializer(_Artifact, _PickleMixin, BaseDataType):
    """Serializer with pickle."""


class PickleEncoder(_Encodable, _PickleMixin, BaseDataType):
    """Pickle inline encoder."""


class _DillMixin:
    def _encode_data(self, item):
        return dill.dumps(item, recurse=True)

    def decode_data(self, item):
        return dill.loads(item)


class DillSerializer(_Artifact, _DillMixin, BaseDataType):
    """Serializer with dill.

    This is also the default serializer.
    >>> from superduper.components.datatype import DEFAULT_SERIALIZER
    """


class _DillEncoder(_Encodable, _DillMixin, BaseDataType):
    ...


class FileType(BaseDataType):
    """Type for encoding files on disk."""

    encodable: t.ClassVar[str] = 'file'

    def encode_data(self, item):
        assert os.path.exists(item)
        return File(path=item)

    def decode_data(self, item):
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
        pass

    @abstractmethod
    def unpack(self):
        pass


class File(Saveable):
    """Placeholder for a file.

    :param path: Path to file.
    """

    path: str = ''

    def __post_init__(self, db=None):
        if not self.identifier:
            self.identifier = get_hash(self.path)
        return super().__post_init__(db)

    def init(self):
        if self.path:
            return
        self.path = self.db.artifact_store.get_file(self.identifier)

    def unpack(self):
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

    def init(self):
        if self.bytes:
            return
        self.bytes = self.db.artifact_store.get_bytes(self.identifier)

    def unpack(self):
        self.init()
        return self.bytes

    @property
    def reference(self):
        return f'&:blob:{self.identifier}'


json_encoder = JSON('json')
pickle_encoder = PickleEncoder('pickle_encoder')
pickle_serializer = PickleSerializer('pickle_serializer')
dill_encoder = _DillEncoder('dill_encoder')
dill_serializer = DillSerializer('dill_serializer')
file = FileType('file')

DEFAULT_ENCODER = PickleEncoder('default_encoder')
DEFAULT_SERIALIZER = DillSerializer('default')


INBUILT_DATATYPES = {
    dt.identifier: dt
    for dt in [
        json_encoder,
        pickle_encoder,
        pickle_serializer,
        dill_encoder,
        dill_serializer,
        file,
        DEFAULT_SERIALIZER,
        DEFAULT_ENCODER,
    ]
}
