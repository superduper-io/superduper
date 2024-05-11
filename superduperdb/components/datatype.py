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
from abc import abstractmethod, abstractstaticmethod

import dill
from overrides import override

from superduperdb import CFG
from superduperdb.backends.base.artifacts import (
    ArtifactSavingError,
    _construct_file_id_from_uri,
)
from superduperdb.base.config import BytesEncoding
from superduperdb.base.leaf import Leaf
from superduperdb.components.component import Component, ensure_initialized
from superduperdb.misc.annotations import public_api
from superduperdb.misc.hash import random_sha1

Decode = t.Callable[[bytes], t.Any]
Encode = t.Callable[[t.Any], bytes]


class IntermidiaType:
    """Intermidia data type."""

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

    from superduperdb.ext.torch.utils import device_of

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
    """Abstract class for creating a DataType."""

    @abstractstaticmethod
    def check(data: t.Any) -> bool:
        """Check if the data can be encoded by the DataType.

        If the data can be encoded, return True, otherwise False

        :param data: The data to check
        """
        raise NotImplementedError

    @abstractstaticmethod
    def create(data: t.Any) -> "DataType":
        """Create a DataType for the data.

        :param data: The data to create the DataType for
        """
        raise NotImplementedError


@public_api(stability='stable')
@dc.dataclass(kw_only=True)
class DataType(Component):
    """A data type component that defines how data is encoded and decoded.

    {component_parameters}

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
    :param intermidia_type: Type of the intermidia data
           [IntermidiaType.BYTES, IntermidiaType.STRING]
    :param media_type: The media type.
    """

    __doc__ = __doc__.format(component_parameters=Component.__doc__)

    ui_schema: t.ClassVar[t.List[t.Dict]] = [
        {
            'name': 'serializer',
            'type': 'string',
            'choices': ['pickle', 'dill', 'torch'],
            'default': 'dill',
        },
        {'name': 'info', 'type': 'json', 'optional': True},
        {'name': 'shape', 'type': 'json', 'optional': True},
        {'name': 'directory', 'type': 'str', 'optional': True},
        {
            'name': 'encodable',
            'type': 'str',
            'choices': ['encodable', 'lazy_artifact', 'file'],
            'default': 'lazy_artifact',
        },
        {
            'name': 'bytes_encoding',
            'type': 'str',
            'choices': ['base64', 'bytes'],
            'default': 'bytes',
        },
        {'name': 'media_type', 'type': 'str', 'optional': True},
    ]

    type_id: t.ClassVar[str] = 'datatype'
    encoder: t.Optional[t.Callable] = None  # not necessary if encodable is file
    decoder: t.Optional[t.Callable] = None
    info: t.Optional[t.Dict] = None
    shape: t.Optional[t.Sequence] = None
    directory: t.Optional[str] = None
    encodable: str = 'encodable'
    bytes_encoding: t.Optional[str] = CFG.bytes_encoding
    intermidia_type: t.Optional[str] = IntermidiaType.BYTES
    media_type: t.Optional[str] = None
    registered_types: t.ClassVar[t.Dict[str, "DataType"]] = {}

    def __post_init__(self, artifacts):
        """Post-initialization hook.

        :param artifacts: The artifacts.
        """
        super().__post_init__(artifacts)
        self.encodable_cls = _ENCODABLES[self.encodable]
        self._takes_x = 'x' in inspect.signature(self.encodable_cls.__init__).parameters
        self.bytes_encoding = self.bytes_encoding or CFG.bytes_encoding
        self.register_datatype(self)

    def dict(self):
        """Get the dictionary representation of the object."""
        r = super().dict()
        if hasattr(self.bytes_encoding, 'value'):
            r['dict']['bytes_encoding'] = str(self.bytes_encoding.value)
        return r

    def __call__(
        self, x: t.Optional[t.Any] = None, uri: t.Optional[str] = None
    ) -> '_BaseEncodable':
        """Create an instance of the encodable class.

        :param x: The optional content.
        :param uri: The optional URI.
        """
        if self._takes_x:
            return self.encodable_cls(datatype=self, x=x, uri=uri)
        else:
            return self.encodable_cls(datatype=self, uri=uri)

    @ensure_initialized
    def encode_data(self, item, info: t.Optional[t.Dict] = None):
        """Encode the item into bytes.

        :param item: The item to encode.
        :param info: The optional information dictionary.
        """
        info = info or {}
        data = self.encoder(item, info)
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
        return self.decoder(item, info=info)

    def bytes_encoding_after_encode(self, data):
        """Encode the data to base64.

        if the bytes_encoding is BASE64 and the intermidia_type is BYTES

        :param data: Encoded data
        """
        if (
            self.bytes_encoding == BytesEncoding.BASE64
            and self.intermidia_type == IntermidiaType.BYTES
        ):
            return bytes_to_base64(data)
        return data

    def bytes_encoding_before_decode(self, data):
        """Encode the data to base64.

        if the bytes_encoding is BASE64 and the intermidia_type is BYTES

        :param data: Decoded data
        """
        if (
            self.bytes_encoding == BytesEncoding.BASE64
            and self.intermidia_type == IntermidiaType.BYTES
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


# TODO: Remove this because this function is only used in test cases.
def build_torch_state_serializer(module, info):
    """Datatype for serializing torch state dict.

    :param module: Module.
    :param info: Information.
    """
    return DataType(
        identifier=module.__name__,
        info=info,
        encoder=encode_torch_state_dict,
        decoder=DecodeTorchStateDict(module),
    )


def _find_descendants(cls):
    """Find descendants of the given class.

    :param cls: The class to find descendants for.
    """
    descendants = cls.__subclasses__()
    for subclass in descendants:
        descendants.extend(_find_descendants(subclass))
    return descendants


@dc.dataclass(kw_only=True)
class _BaseEncodable(Leaf):
    """Data variable wrapping encode-able item.

    Encoding is controlled by the referred
    to ``Encoder`` instance.

    :param encoder: Instance of ``Encoder`` controlling encoding.
    :param x: Wrapped content.
    :param uri: URI of the content, if any.
    """

    file_id: t.Optional[str] = None
    datatype: DataType
    uri: t.Optional[str] = None
    sha1: t.Optional[str] = None

    def _deep_flat_encode(self, cache):
        """Deep flat encode the encodable item.

        :param cache: Cache to store encoded items.
        """
        r = self.encode()
        out = {
            'cls': self.__class__.__name__,
            'module': self.__class__.__module__,
            'dict': {
                'file_id': r['_content']['file_id'],
                'bytes': r['_content']['bytes'],
                'uri': r['_content']['uri'],
                'datatype': f'_component/datatype/{r["_content"]["datatype"]}',
            },
            'id': f'_{self.leaf_type}/{r["_content"]["file_id"]}',
        }
        cache[out['id']] = out
        return out['id']

    @property
    def id(self):
        """Get the ID of the encodable item."""
        assert self.file_id is not None
        return f'_{self.leaf_type}/{self.file_id}'

    def __post_init__(self):
        """Post-initialization hook."""
        if self.uri is not None and self.file_id is None:
            self.file_id = _construct_file_id_from_uri(self.uri)

        if self.uri and not re.match('^[a-z]{0,5}://', self.uri):
            self.uri = f'file://{self.uri}'

    @property
    def unique_id(self):
        """Get the unique ID of the encodable item."""
        if self.file_id is not None:
            return self.file_id
        return str(id(self.x))

    @property
    def reference(self):
        """Get the reference to the datatype."""
        return self.datatype.reference

    def unpack(self, db):
        """Unpack the content of the `Encodable`.

        :param db: Datalayer instance.
        """
        return self.x

    @classmethod
    def get_encodable_cls(cls, name, default=None):
        """Get the subclass of the _BaseEncodable with the given name.

        All the registered subclasses must be subclasses of the _BaseEncodable.

        :param name: Name of the subclass.
        :param default: Default class to return if no subclass is found.
        :raises ValueError: If no subclass with the name is found and no default
                            is provided.
        """
        for sub_cls in _find_descendants(cls):
            if sub_cls.__name__.lower() == name.lower().replace('_', '').replace(
                '-', ''
            ):
                return sub_cls
        if default is None:
            raise ValueError(f'No subclass with name "{name}" found.')
        elif not issubclass(default, cls):
            raise ValueError(
                "The default class must be a subclass of the _BaseEncodable."
            )
        return default

    @classmethod
    @abstractmethod
    def _get_object(cls, db, r):
        """Get object from the given representation.

        :param db: Datalayer instance.
        :param r: Representation of the object.
        """
        pass

    @classmethod
    @abstractmethod
    def decode(cls, r, db=None) -> '_BaseEncodable':
        """Decode the representation to an instance of _BaseEncodable.

        :param r: Representation to decode.
        :param db: Datalayer instance.
        """
        pass

    def get_hash(self, data):
        """Get the hash of the given data.

        :param data: Data to hash.
        """
        if isinstance(data, str):
            bytes_ = data.encode()
        elif isinstance(data, bytes):
            bytes_ = data
        else:
            raise ValueError(f'Unsupported data type: {type(data)}')

        return hashlib.sha1(bytes_).hexdigest()


class Empty:
    """Sentinel class."""

    def __repr__(self):
        """Get the string representation of the Empty object."""
        return '<EMPTY>'


@dc.dataclass
class Encodable(_BaseEncodable):
    """Class for encoding non-Python datatypes to the database.

    :param x: The encodable object.
    """

    x: t.Any = Empty()
    leaf_type: t.ClassVar[str] = 'encodable'

    def _encode(self):
        bytes_ = self.datatype.encode_data(self.x)
        sha1 = self.get_hash(bytes_)
        return bytes_, sha1

    @classmethod
    def _get_object(cls, db, r):
        if r.get('bytes') is None:
            return None
        datatype = cls.get_datatype(db, r)
        object = datatype.decode_data(r['bytes'], info=datatype.info)
        return object

    @override
    def encode(self, leaf_types_to_keep: t.Sequence = ()):
        """Encode itself to a specific format.

        Encode `self.x` to dictionary format which could be serialized
        to a database.

        :param leaf_types_to_keep: Leaf nodes to keep from encoding.
        """
        bytes_, sha1 = self._encode()
        return {
            '_content': {
                'bytes': bytes_,
                'datatype': self.datatype.identifier,
                'leaf_type': self.leaf_type,
                'sha1': sha1,
                'uri': self.uri,
                'file_id': sha1 if self.file_id is None else self.file_id,
                'id': (
                    f'_{self.leaf_type}/'
                    f'{sha1 if self.file_id is None else self.file_id}',
                ),
            }
        }

    @classmethod
    def build(cls, r):
        """Build an `Encodable` instance with the given parameters `r`.

        :param r: Parameters for building the `Encodable` instance.
        """
        return cls(**r)

    def init(self, db):
        """Initialization method.

        :param db: The Datalayer instance.
        """
        pass

    @classmethod
    def decode(cls, r, db=None) -> 'Encodable':
        """Decode the dictionary `r` to an `Encodable` instance.

        :param r: The dictionary to decode.
        :param db: The Datalayer instance.
        """
        object = cls._get_object(db, r['_content'])
        return cls(
            x=object,
            datatype=cls.get_datatype(db, r['_content']),
            uri=r['_content']['uri'],
            file_id=r['_content'].get('file_id'),
        )

    @classmethod
    def get_datatype(cls, db, r):
        """Get the datatype of the object.

        :param db: `Datalayer` instance to assist with
        :param r: The object to get the datatype from
        """
        if db is None:
            try:
                from superduperdb.components.datatype import serializers

                datatype = serializers[r['datatype']]
            except KeyError:
                raise ValueError(
                    f'You specified a serializer which doesn\'t have a'
                    f' default value: {r["datatype"]}'
                )
        else:
            datatype = db.datatypes[r['datatype']]
        return datatype


@dc.dataclass
class Native(_BaseEncodable):
    """Class for representing native data supported by the underlying database.

    :param x: The encodable object.
    """

    leaf_type: t.ClassVar[str] = 'native'
    x: t.Optional[t.Any] = None

    @classmethod
    def _get_object(cls, db, r):
        raise NotImplementedError

    @override
    def encode(self, leaf_types_to_keep: t.Sequence = ()):
        """Encode itself to a specific format.

        :param leaf_types_to_keep: Leaf nodes to keep from encoding.
        """
        return self.x

    @classmethod
    def decode(cls, r, db=None):
        """Decode the object `r` to a `Native` instance.

        :param r: The object to decode.
        :param db: The Datalayer instance.
        """
        return r


class _ArtifactSaveMixin:
    def save(self, artifact_store):
        r = artifact_store.save_artifact(self.encode()['_content'])
        self.x = None
        self.file_id = r['file_id']


@dc.dataclass
class Artifact(_BaseEncodable, _ArtifactSaveMixin):
    """Class for representing data to be saved on disk or in the artifact-store.

    :param x: The artifact object.
    :param artifact: Whether the object is an artifact.
    """

    leaf_type: t.ClassVar[str] = 'artifact'
    x: t.Any = Empty()
    artifact: bool = False

    def _encode(self):
        bytes_ = self.datatype.encode_data(self.x)
        sha1 = self.get_hash(bytes_)
        return bytes_, sha1

    def init(self, db):
        """Initialize the x attribute with the actual value from the artifact store.

        :param db: The Datalayer instance.
        """
        if isinstance(self.x, Empty):
            self.x = self._get_object(
                db,
                file_id=self.file_id,
                datatype=self.datatype.identifier,
                uri=self.uri,
            )

    @override
    def encode(self, leaf_types_to_keep: t.Sequence = ()):
        """Encode itself to a specific format.

        Encode `self.x` to dictionary format which is later saved
        in an artifact store.

        :param leaf_types_to_keep: Leaf nodes to exclude.
        """
        if self.x is not None and not isinstance(self.x, Empty):
            bytes_, sha1 = self._encode()
        else:
            bytes_, sha1 = None, None
        return {
            '_content': {
                'bytes': bytes_,
                'datatype': self.datatype.identifier,
                'leaf_type': self.leaf_type,
                'sha1': sha1,
                'uri': self.uri,
                'file_id': sha1 if self.file_id is None else self.file_id,
                'id': (
                    f'_{self.leaf_type}/'
                    f'{sha1 if self.file_id is None else self.file_id}',
                ),
                'artifact_type': 'bytes',
            }
        }

    @classmethod
    def _get_object(cls, db, file_id, datatype, uri):
        obj = db.artifact_store.load_artifact(
            {
                'file_id': file_id,
                'datatype': datatype,
                'uri': uri,
            }
        )
        obj = db.datatypes[datatype].bytes_encoding_before_decode(obj)
        return obj

    def unpack(self, db):
        """Unpack the content of the `Encodable`.

        :param db: The `Datalayer` instance to assist with unpacking.
        """
        self.init(db=db)
        return self.x

    def save(self, artifact_store):
        """Save the encoded data into an artifact store.

        :param artifact_store: The artifact store for storing the encoded object.
        """
        r = artifact_store.save_artifact(self.encode()['_content'])
        self.x = None
        self.file_id = r['file_id']

    @classmethod
    def decode(cls, r, db=None) -> 'Artifact':
        """Decode the dictionary `r` into an instance of `Artifact`.

        :param r: The dictionary to decode.
        :param db: The Datalayer instance.
        """
        r = r['_content']
        x = cls._get_object(
            db, file_id=r['file_id'], datatype=r['datatype'], uri=r.get('uri')
        )
        return cls(
            x=x,
            datatype=db.datatypes[r['datatype']],
            uri=r.get('uri'),
            file_id=r.get('file_id'),
            sha1=r.get('sha1'),
        )


@dc.dataclass
class LazyArtifact(Artifact):
    """A class for loading an artifact only when needed.

    :param artifact: If the object is an artifact
    """

    leaf_type: t.ClassVar[str] = 'lazy_artifact'
    artifact: bool = False

    def __post_init__(self):
        if self.datatype.bytes_encoding == BytesEncoding.BASE64:
            raise ArtifactSavingError('BASE64 is not supported on disk!')

    @override
    def encode(self, leaf_types_to_keep: t.Sequence = ()):
        """Encode `x` in dictionary format for the artifact store.

        :param leaf_types_to_keep: Leaf nodes to exclude
                                   from encoding.
        """
        return super().encode(leaf_types_to_keep)

    def unpack(self, db):
        """Unpack the content of the `Encodable`.

        :param db: `Datalayer` instance to assist with
        """
        self.init(db=db)
        return self.x

    def save(self, artifact_store):
        """Save the encoded data into the artifact store.

        :param artifact_store: Artifact store for saving
                               the encoded object.
        """
        r = artifact_store.save_artifact(self.encode()['_content'])
        self.x = None
        self.file_id = r['file_id']

    @classmethod
    def decode(cls, r, db=None) -> 'LazyArtifact':
        """Decode data into a `LazyArtifact` instance.

        :param r: Object to decode
        :param db: Datalayer instance
        """
        return cls(
            x=Empty(),  # this is to enable lazy loading
            datatype=db.datatypes[r['_content']['datatype']],
            uri=r['_content']['uri'],
            file_id=r['_content'].get('file_id'),
        )


@dc.dataclass
class File(_BaseEncodable, _ArtifactSaveMixin):
    """Data to be saved on disk and passed as a file reference.

    :param x: File object
    """

    leaf_type: t.ClassVar[str] = 'file'
    x: t.Any = Empty()

    def init(self, db):
        """Initialize to load `x` with the actual file from the artifact store.

        :param db: A Datalayer instance
        """
        if isinstance(self.x, Empty):
            file = db.artifact_store._load_file(self.file_id)
            self.x = file

    def unpack(self, db):
        """Unpack and get the original data.

        :param db: Datalayer instance.
        """
        self.init(db)
        return self.x

    @classmethod
    def _get_object(cls, db, r):
        return r['x']

    @override
    def encode(self, leaf_types_to_keep: t.Sequence = ()):
        """Encode `x` to a dictionary which is saved to the artifact store later.

        :param leaf_types_to_keep: Leaf nodes to exclude
                                   from encoding
        """
        dc.asdict(self)
        file_id = self.file_id or random_sha1()
        return {
            '_content': {
                'datatype': self.datatype.identifier,
                'leaf_type': self.leaf_type,
                'uri': self.x,
                'file_id': file_id,
                'id': f'_{self.leaf_type}/{file_id}',
                'artifact_type': 'file',
            }
        }

    @classmethod
    def decode(cls, r, db=None) -> 'File':
        """Decode data to a `File` instance.

        :param r: Object to decode
        :param db: Datalayer instance
        """
        file = cls(
            x=Empty(),
            datatype=db.datatypes[r['_content']['datatype']],
            file_id=r['_content']['file_id'],
        )
        file.init(db)
        return file


class LazyFile(File):
    """Class is used to load a file only when needed."""

    leaf_type: t.ClassVar[str] = 'lazy_file'

    @classmethod
    def decode(cls, r, db=None) -> 'LazyFile':
        """Decode a dictionary to a `LazyFile` instance.

        :param r: Object to decode
        :param db: Datalayer instance
        """
        file = cls(
            x=Empty(),
            datatype=db.datatypes[r['_content']['datatype']],
            file_id=r['_content']['file_id'],
        )
        return file


Encoder = DataType


_ENCODABLES = {
    'encodable': Encodable,
    'artifact': Artifact,
    'file': File,
    'native': Native,
    'lazy_artifact': LazyArtifact,
    'lazy_file': LazyFile,
}

json_serializer = DataType(
    'json',
    encoder=json_encode,
    decoder=json_decode,
    encodable='encodable',
    bytes_encoding=BytesEncoding.BASE64,
    intermidia_type=IntermidiaType.STRING,
)

pickle_serializer = DataType(
    'pickle',
    encoder=pickle_encode,
    decoder=pickle_decode,
    encodable='artifact',
)
pickle_lazy = DataType(
    'pickle_lazy',
    encoder=pickle_encode,
    decoder=pickle_decode,
    encodable='lazy_artifact',
)
dill_serializer = DataType(
    'dill',
    encoder=dill_encode,
    decoder=dill_decode,
    encodable='artifact',
)
dill_lazy = DataType(
    'dill_lazy',
    encoder=dill_encode,
    decoder=dill_decode,
    encodable='lazy_artifact',
)
torch_serializer = DataType(
    'torch',
    encoder=torch_encode,
    decoder=torch_decode,
    encodable='lazy_artifact',
)
file_serializer = DataType(
    'file',
    encoder=file_check,
    decoder=file_check,
    encodable="file",
)
file_lazy = DataType(
    'file_lazy',
    encoder=file_check,
    decoder=file_check,
    encodable="lazy_file",
)
serializers = DataType.registered_types
