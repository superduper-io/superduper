import base64
import dataclasses as dc
import hashlib
import inspect
import io
import json
import os
import pickle
import re
import typing as t
from abc import abstractmethod

import dill

from superduper import CFG
from superduper.backends.base.artifacts import (
    _construct_file_id_from_uri,
)
from superduper.base.config import BytesEncoding
from superduper.base.leaf import Leaf
from superduper.components.component import Component, ensure_initialized
from superduper.misc.annotations import component
from superduper.misc.hash import hash_path

Decode = t.Callable[[bytes], t.Any]
Encode = t.Callable[[t.Any], bytes]

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


class IntermediateType:
    """Intermediate data type # noqa."""

    BYTES = 'bytes'
    STRING = 'string'


def json_encode(object: t.Any, info: t.Optional[t.Dict] = None) -> str:
    """Encode the dict to a JSON string.

    :param object: The object to encode
    :param info: Optional information
    """
    return json.dumps(object)


def json_decode(b: str, info: t.Optional[t.Dict] = None) -> t.Any:
    """Decode the JSON string to an dict.

    :param b: The JSON string to decode
    :param info: Optional information
    """
    return json.loads(b)


def pickle_encode(object: t.Any, info: t.Optional[t.Dict] = None) -> bytes:
    """Encodes an object using pickle.

    :param object: The object to encode.
    :param info: Optional information.
    """
    return pickle.dumps(object)


def pickle_decode(b: bytes, info: t.Optional[t.Dict] = None) -> t.Any:
    """Decodes bytes using pickle.

    :param b: The bytes to decode.
    :param info: Optional information.
    """
    return pickle.loads(b)


def dill_encode(object: t.Any, info: t.Optional[t.Dict] = None) -> bytes:
    """Encodes an object using dill.

    :param object: The object to encode.
    :param info: Optional information.
    """
    return dill.dumps(object, recurse=True)


def dill_decode(b: bytes, info: t.Optional[t.Dict] = None) -> t.Any:
    """Decodes bytes using dill.

    :param b: The bytes to decode.
    :param info: Optional information.
    """
    return dill.loads(b)


def file_check(path: t.Any, info: t.Optional[t.Dict] = None) -> str:
    """Checks if a file path exists.

    :param path: The file path to check.
    :param info: Optional information.
    :raises ValueError: If the path does not exist.
    """
    if not (isinstance(path, str) and os.path.exists(path)):
        raise ValueError(f"Path '{path}' does not exist")
    return path


def torch_encode(object: t.Any, info: t.Optional[t.Dict] = None) -> bytes:
    """Saves an object in torch format.

    :param object: The object to encode.
    :param info: Optional information.
    """
    import torch

    from superduper.ext.torch.utils import device_of

    if not isinstance(object, dict):
        previous_device = str(device_of(object))
        object.to('cpu')
        f = io.BytesIO()
        torch.save(object, f)
        object.to(previous_device)
    else:
        f = io.BytesIO()
        torch.save(object, f)

    return f.getvalue()


def torch_decode(b: bytes, info: t.Optional[t.Dict] = None) -> t.Any:
    """Decodes bytes to a torch model.

    :param b: The bytes to decode.
    :param info: Optional information.
    """
    import torch

    return torch.load(io.BytesIO(b))


def bytes_to_base64(bytes):
    """Converts bytes to base64.

    :param bytes: The bytes to convert.
    """
    return base64.b64encode(bytes).decode('utf-8')


def base64_to_bytes(encoded):
    """Decodes a base64 encoded string.

    :param encoded: The base64 encoded string.
    """
    return base64.b64decode(encoded)


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
    def create(data: t.Any) -> "DataType":
        """Create a DataType for the data.

        :param data: The data to create the DataType for
        """
        raise NotImplementedError


class DataType(Component):
    """A data type component that defines how data is encoded and decoded.

    :param encoder: A callable that converts an encodable object of this
                    encoder to bytes.
    :param decoder: A callable that converts bytes to an encodable object
                    of this encoder.
    :param info: An optional information dictionary.
    :param shape: The shape of the data.
    :param directory: The directory to store file types.
    :param encodable: The type of encodable object ('encodable',
                      'lazy_artifact', or 'file').
    :param bytes_encoding: The encoding type for bytes ('base64' or 'bytes').
    :param intermediate_type: Type of the intermediate data
           [IntermediateType.BYTES, IntermediateType.STRING]
    :param media_type: The media type.
    """

    type_id: t.ClassVar[str] = 'datatype'
    encoder: t.Optional[t.Callable] = None  # not necessary if encodable is file
    decoder: t.Optional[t.Callable] = None
    info: t.Optional[t.Dict] = None
    shape: t.Optional[t.Sequence] = None
    directory: t.Optional[str] = None
    encodable: str = 'encodable'
    bytes_encoding: t.Optional[str] = CFG.bytes_encoding
    intermediate_type: t.Optional[str] = IntermediateType.BYTES
    media_type: t.Optional[str] = None
    registered_types: t.ClassVar[t.Dict[str, "DataType"]] = {}
    cache: bool = True

    def __post_init__(self, db, artifacts):
        """Post-initialization hook.

        :param artifacts: The artifacts.
        """
        super().__post_init__(db, artifacts)
        if self.encodable in _ENCODABLES:
            self.encodable_cls = _ENCODABLES[self.encodable]
        else:
            import importlib

            self.encodable_cls = importlib.import_module(
                '.'.join(self.encodable.split('.')[:-1])
            ).__dict__[self.encodable.split('.')[-1]]

        self.bytes_encoding = self.bytes_encoding or CFG.bytes_encoding
        self.register_datatype(self)

    @property
    def artifact(self):
        """Check if the encodable is an artifact."""
        return self.encodable_cls.artifact

    def dict(self, metadata: bool = True, defaults: bool = True):
        """Get the dictionary representation of the object."""
        r = super().dict(metadata=metadata, defaults=defaults)
        if hasattr(self.bytes_encoding, 'value'):
            r['bytes_encoding'] = str(self.bytes_encoding.value)  # type: ignore[union-attr]
        return r

    def __call__(
        self, x: t.Optional[t.Any] = None, uri: t.Optional[str] = None
    ) -> '_BaseEncodable':
        """Create an instance of the encodable class.

        :param x: The optional content.
        :param uri: The optional URI.
        """
        return self.encodable_cls(datatype=self, x=x, uri=uri, db=self.db)

    @ensure_initialized
    def encode_data_with_identifier(self, item, info: t.Optional[t.Dict] = None):
        """Encode the item into bytes.

        :param item: The item to encode.
        :param info: The optional information dictionary.
        """
        info = info or {}
        data = self.encoder(item, info) if self.encoder else item
        sha1 = self.encodable_cls.get_hash(data)
        data = self.bytes_encoding_after_encode(data)
        return data, sha1

    @ensure_initialized
    def encode_data(self, item, info: t.Optional[t.Dict] = None):
        """Encode the item into bytes.

        :param item: The item to encode.
        :param info: The optional information dictionary.
        """
        info = info or {}
        data = self.encoder(item, info) if self.encoder else item
        data = self.bytes_encoding_after_encode(data)
        return data

    @ensure_initialized
    def decode_data(self, item, info: t.Optional[t.Dict] = None):
        """Decode the item from bytes.

        :param item: The item to decode.
        :param info: The optional information dictionary.
        """
        info = info or {}
        item = self.bytes_encoding_before_decode(item)
        return self.decoder(item, info=info) if self.decoder else item

    def bytes_encoding_after_encode(self, data):
        """Encode the data to base64.

        if the bytes_encoding is BASE64 and the intermediate_type is BYTES

        :param data: Encoded data
        """
        if (
            self.bytes_encoding == BytesEncoding.BASE64
            and self.intermediate_type == IntermediateType.BYTES
        ):
            return bytes_to_base64(data)
        return data

    def bytes_encoding_before_decode(self, data):
        """Encode the data to base64.

        if the bytes_encoding is BASE64 and the intermediate_type is BYTES

        :param data: Decoded data
        """
        if (
            self.bytes_encoding == BytesEncoding.BASE64
            and self.intermediate_type == IntermediateType.BYTES
        ):
            return base64_to_bytes(data)
        return data

    @classmethod
    def register_datatype(cls, instance):
        """Register a datatype.

        :param instance: The datatype instance to register.
        """
        cls.registered_types[instance.identifier] = instance


def encode_torch_state_dict(module, info):
    """Encode torch state dictionary.

    :param module: Module.
    :param info: Information.
    """
    import torch

    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)

    return buffer.getvalue()


class DecodeTorchStateDict:
    """Torch state dictionary decoder.

    :param cls: Torch state cls
    """

    def __init__(self, cls):
        self.cls = cls

    def __call__(self, b: bytes, info: t.Dict):
        """Decode the torch state dictionary.

        :param b: Bytes.
        :param info: Information.
        """
        import torch

        buffer = io.BytesIO(b)
        module = self.cls(**info)
        module.load_state_dict(torch.load(buffer))
        return module


def _find_descendants(cls):
    """Find descendants of the given class.

    :param cls: The class to find descendants for.
    """
    descendants = cls.__subclasses__()
    for subclass in descendants:
        descendants.extend(_find_descendants(subclass))
    return descendants


class _BaseEncodable(Leaf):
    """Data variable wrapping encode-able item.

    Encoding is controlled by the referred
    to ``Encoder`` instance.

    :param datatype: The datatype of the content.
    :param uri: URI of the content, if any.
    :param x: Wrapped content.
    """

    identifier: str = ''
    datatype: DataType
    uri: t.Optional[str] = None  # URI of the content to be deprecated
    x: t.Optional[t.Any] = None
    lazy: t.ClassVar[bool] = False
    artifact: t.ClassVar[bool] = False

    def __post_init__(self, db):
        """Post-initialization hook.

        :param db: Datalayer instance.
        """
        db = db or self.datatype.db
        super().__post_init__(db)
        if self.uri is not None and self.identifier is None:
            self.identifier = _construct_file_id_from_uri(self.uri)

        if self.uri and not re.match('^[a-z]{0,5}://', self.uri):
            self.uri = f'file://{self.uri}'

    @property
    def reference(self):
        """Get the reference to the datatype."""
        return self.datatype.reference

    def unpack(self):
        """Unpack the content of the `Encodable`."""
        return self.x

    @staticmethod
    def get_hash(data):
        """Get the hash of the given data.

        :param data: Data to hash.
        """
        if isinstance(data, str):
            bytes_ = data.encode()
        elif isinstance(data, bytes):
            bytes_ = data
        elif isinstance(data, Native):
            bytes_ = str([type(data), data.x]).encode()
        else:
            bytes_ = str(id(data)).encode()
        return hashlib.sha1(bytes_).hexdigest()

    @staticmethod
    def build_reference(identifier, source_data):
        raise NotImplementedError


class Empty:
    """Sentinel class # noqa."""

    def __repr__(self):
        """Get the string representation of the Empty object."""
        return '<EMPTY>'


class Blob(Leaf):
    """A wrapper to signify a blob for special treatment.

    See `Document.encode` and related functions.

    :param identifier: The identifier of the blob.
    :param bytes: The bytes of the blob.
    """

    identifier: str
    bytes: bytes


class Encodable(_BaseEncodable):
    """Class for encoding non-Python datatypes to the database.

    :param x: The encodable object.
    :param blob: The blob data.
    """

    x: t.Any = Empty()
    artifact: t.ClassVar[bool] = False
    blob: dc.InitVar[t.Optional[bytearray]] = None
    leaf_type: t.ClassVar[str] = 'encodable'

    def __post_init__(self, db, blob):
        super().__post_init__(db)
        if isinstance(self.x, Empty):
            self.datatype.init()
            self.x = self.datatype.decode_data(blob)

    def _encode(self):
        bytes_ = self.datatype.encode_data(self.x)
        sha1 = self.get_hash(bytes_)
        return bytes_, sha1

    def to_artifact(self):
        """Convert the encodable to an artifact."""
        r = self.dict()
        r['datatype'].encodable = 'artifact'
        kwargs = {
            k: v for k, v in r.items() if k in inspect.signature(Artifact).parameters
        }
        return Artifact(**kwargs)

    def dict(self, metadata: bool = True, defaults: bool = True):
        """Get the dictionary representation of the object."""
        r = super().dict(metadata=metadata, defaults=defaults)
        del r['x']
        r['blob'], identifier = self._encode()
        if not r['identifier']:
            self.identifier = identifier
            r['identifier'] = identifier
        return r

    def init(self, db):
        """Initialization method.

        :param db: The Datalayer instance.
        """
        pass

    @classmethod
    def get_datatype(cls, db, r):
        """Get the datatype of the object.

        :param db: `Datalayer` instance to assist with
        :param r: The object to get the datatype from
        """
        if db is None:
            try:
                from superduper.components.datatype import serializers

                datatype = serializers[r['datatype']]
            except KeyError:
                raise ValueError(
                    f'You specified a serializer which doesn\'t have a'
                    f' default value: {r["datatype"]}'
                )
        else:
            datatype = db.datatypes[r['datatype']]
        return datatype


class Native(_BaseEncodable):
    """Class for representing native data supported by the underlying database.

    :param x: The encodable object.
    """

    leaf_type: t.ClassVar[str] = 'native'
    x: t.Optional[t.Any] = None

    def __post_init__(self, db):
        return super().__post_init__(db)

    @classmethod
    def _get_object(cls, db, r):
        raise NotImplementedError


class Artifact(_BaseEncodable):
    """Class for representing data to be saved on disk or in the artifact-store.

    :param x: The artifact object.
    :param blob: The blob data. Can be a string or bytes.
                if string, it should be in the format `&:blob:{file_id}`
                if bytes, it should be the actual data.
    """

    leaf_type: t.ClassVar[str] = 'artifact'
    artifact: t.ClassVar[bool] = True
    x: t.Any = Empty()
    blob: dc.InitVar[t.Optional[t.Union[str, bytes]]] = None
    lazy: t.ClassVar[bool] = False

    def __post_init__(self, db, blob=None):
        super().__post_init__(db)
        self._blob = blob
        self._reference = None

        if not (self.lazy and not isinstance(self._blob, bytes)):
            self.init()

    def init(self, db=None):
        """Initialize to load `x` with the actual file from the artifact store."""
        if isinstance(self._blob, t.Callable):
            self._blob, _ = self._blob()

        if isinstance(self._blob, bytes):
            blob = self._blob
            self.datatype.init()
            self.x = self.datatype.decoder(blob, info=None)
            self._blob = None

        if not isinstance(self.x, Empty):
            return

    def dict(self, metadata: bool = True, defaults: bool = True):
        """Get the dictionary representation of the object."""
        bytes, identifier = self._encode()
        if not self.identifier:
            self.identifier = identifier
        r = super().dict(metadata=metadata, defaults=defaults)
        del r['x']
        r['blob'] = Blob(identifier=self.identifier, bytes=bytes)
        return r

    def _encode(self):
        bytes_ = self.datatype.encoder(self.x)
        sha1 = self.get_hash(bytes_)
        return bytes_, sha1

    def unpack(self):
        """Unpack the content of the `Encodable`."""
        self.init()
        return self.x

    @staticmethod
    def build_reference(identifier, source_data):
        """Build a reference to the blob.

        :param identifier: The identifier of the blob.
        :param source_data: The source data.
        :return: The reference to the blob. '&:blob:{file_id}'
        """
        return f"&:blob:{identifier}"


class LazyArtifact(Artifact):
    """Data to be saved and loaded only when needed."""

    leaf_type: t.ClassVar[str] = 'lazy_artifact'
    lazy: t.ClassVar[bool] = True

    def dict(self, metadata: bool = True, defaults: bool = True):
        """Get the dictionary representation of the object."""
        self.init()
        return super().dict(metadata=metadata, defaults=defaults)


class FileItem(Leaf):
    """File item class.

    :param identifier: The identifier of the file.
    :param path: The path of the file.
    """

    identifier: str
    path: str


class File(_BaseEncodable):
    """Data to be saved on disk and passed as a file reference.

    :param x: path to the file
    """

    lazy: t.ClassVar[bool] = False
    leaf_type: t.ClassVar[str] = 'file'
    artifact: t.ClassVar[bool] = True

    x: t.Any = Empty()

    def __post_init__(self, db):
        super().__post_init__(db)
        if isinstance(self.x, t.Callable):
            self._file = self.x
            self.x = Empty()
        else:
            self._file = None

        if not self.lazy:
            self.init()

    def init(self, db=None):
        """Initialize to load `x` with the actual file from the artifact store."""
        if isinstance(self._file, t.Callable):
            file_path, self.identifier = self._file()
            self.x = file_path

        if not isinstance(self.x, Empty):
            return

    def dict(self, metadata: bool = True, defaults: bool = True):
        """Get the dictionary representation of the object."""
        self.identifier = self.identifier or hash_path(self.x)
        r = super().dict(metadata=metadata, defaults=defaults)
        r['x'] = FileItem(identifier=self.identifier, path=self.x)
        return r

    def unpack(self):
        """Unpack and get the original data."""
        self.init()
        return self.x

    @staticmethod
    def build_reference(identifier, source_data):
        """Build a reference to the file.

        :param identifier: The identifier of the file.
        :param source_data: The source data.
        :return: The reference to the file. '?:file:{file_id}'
        """
        return f"&:file:{identifier}"


class LazyFile(File):
    """Class is used to load a file only when needed."""

    leaf_type: t.ClassVar[str] = 'lazy_file'
    lazy: t.ClassVar[bool] = True

    def dict(self, metadata: bool = True, defaults: bool = True):
        """Get the dictionary representation of the object."""
        self.init()
        return super().dict(metadata=metadata, defaults=defaults)


_ENCODABLES = {
    'encodable': Encodable,
    'artifact': Artifact,
    'lazy_artifact': LazyArtifact,
    'file': File,
    'native': Native,
    'lazy_file': LazyFile,
}


methods: t.Dict[str, t.Dict] = {
    'pickle': {'encoder': pickle_encode, 'decoder': pickle_decode},
    'dill': {'encoder': dill_encode, 'decoder': dill_decode},
    'torch': {'encoder': torch_encode, 'decoder': torch_decode},
    'file': {'encoder': file_check, 'decoder': file_check},
    'native': {'encoder': None, 'decoder': None},
}


@component()
def get_serializer(
    identifier: str,
    method: str,
    encodable: str = "encodable",
    db: t.Optional['Datalayer'] = None,
):
    """Get a serializer.

    :param identifier: The identifier of the serializer.
    :param method: The method of the serializer.
    :param encodable: The type of encodable object.
    :param db: The Datalayer instance.
    """
    return DataType(
        identifier=identifier,
        encodable=encodable,
        db=db,
        **methods[method],
    )


json_serializer = DataType(
    'json',
    encoder=json_encode,
    decoder=json_decode,
    encodable='encodable',
    bytes_encoding=BytesEncoding.BASE64,
    intermediate_type=IntermediateType.STRING,
)


pickle_encoder = get_serializer(
    identifier='pickle_encoder',
    method='pickle',
    encodable='encodable',
)


pickle_serializer = get_serializer(
    identifier='pickle',
    method='pickle',
    encodable='artifact',
)

pickle_lazy = get_serializer(
    identifier='pickle_lazy',
    method='pickle',
    encodable='lazy_artifact',
)

dill_serializer = get_serializer(
    identifier='dill',
    method='dill',
    encodable='artifact',
)

dill_lazy = get_serializer(
    identifier='dill_lazy',
    method='dill',
    encodable='lazy_artifact',
)

torch_serializer = get_serializer(
    identifier='torch',
    method='torch',
    encodable='lazy_artifact',
)

file_serializer = get_serializer(
    identifier='file',
    method='file',
    encodable='file',
)

file_lazy = get_serializer(
    identifier='file_lazy',
    method='file',
    encodable='lazy_file',
)

serializers = {
    'pickle': pickle_serializer,
    'dill': dill_serializer,
    'torch': torch_serializer,
    'file': file_serializer,
    'pickle_lazy': pickle_lazy,
    'dill_lazy': dill_lazy,
    'file_lazy': file_lazy,
}
