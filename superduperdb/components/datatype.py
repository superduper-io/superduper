import base64
import dataclasses as dc
import hashlib
import inspect
import io
import os
import pickle
import re
import typing as t
from abc import abstractmethod

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


def pickle_encode(object: t.Any, info: t.Optional[t.Dict] = None) -> bytes:
    """
    Encode an object using pickle.

    :param object: Object to encode.
    :param info: Optional information.
    """
    return pickle.dumps(object)


def pickle_decode(b: bytes, info: t.Optional[t.Dict] = None) -> t.Any:
    """
    Decode bytes using pickle.

    :param b: Bytes to decode.
    :param info: Optional information.
    """
    return pickle.loads(b)


def dill_encode(object: t.Any, info: t.Optional[t.Dict] = None) -> bytes:
    """
    Encode an object using dill.

    :param object: Object to encode.
    :param info: Optional information.
    """
    return dill.dumps(object, recurse=True)


def dill_decode(b: bytes, info: t.Optional[t.Dict] = None) -> t.Any:
    """
    Decode bytes using dill.

    :param b: Bytes to decode.
    :param info: Optional information.
    """
    return dill.loads(b)


def file_check(path: t.Any, info: t.Optional[t.Dict] = None) -> str:
    """
    Check if a file path exists.

    :param path: File path to check.
    :param info: Optional information.
    :raises ValueError: If the path does not exist.
    """
    if not (isinstance(path, str) and os.path.exists(path)):
        raise ValueError(f"Path '{path}' does not exist")
    return path


def torch_encode(object: t.Any, info: t.Optional[t.Dict] = None) -> bytes:
    """
    Save an object in torch format.

    :param object: Object to encode.
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
    """
    Decode bytes to torch model.

    :param b: Bytes to decode.
    :param info: Optional information.
    """
    import torch

    return torch.load(io.BytesIO(b))


def to_base64(bytes):
    """
    Convert bytes to base64.

    :param bytes: Bytes to convert.
    """
    return base64.b64encode(bytes).decode('utf-8')


def from_base64(encoded):
    """
    Decode base64 encoded string.

    :param encoded: Base64 encoded string.
    """
    return base64.b64decode(encoded)


@public_api(stability='stable')
@dc.dataclass(kw_only=True)
class DataType(Component):
    """
    {component_parameters}

    :param encoder: Callable converting an ``Encodable`` of this ``Encoder``
    :param decoder: Callable converting a ``bytes`` string to a ``Encodable``
                    of this ``Encoder``.
                    to ``bytes``.
    :param info:  Optional information dictionary
    :param shape: Shape of the data.
    :param directory: Directory to store file types
    :param encodable: 'encodable' or 'file'.
    :param bytes_encoding: bytes test encoding type
    :param media_type: Media type
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
    media_type: t.Optional[str] = None

    def __post_init__(self, artifacts):
        """
        Post-initialization hook.

        :param artifacts: Artifacts.
        """
        super().__post_init__(artifacts)
        self.encodable_cls = _ENCODABLES[self.encodable]
        self._takes_x = 'x' in inspect.signature(self.encodable_cls.__init__).parameters
        self.bytes_encoding = self.bytes_encoding or CFG.bytes_encoding

    def dict(self):
        """
        Get dictionary representation.

        """
        r = super().dict()
        if hasattr(self.bytes_encoding, 'value'):
            r['dict']['bytes_encoding'] = str(self.bytes_encoding.value)
        return r

    def __call__(
        self, x: t.Optional[t.Any] = None, uri: t.Optional[str] = None
    ) -> '_BaseEncodable':
        """
        Call method.

        :param x: Optional content.
        :param uri: Optional URI.
        """
        if self._takes_x:
            return self.encodable_cls(datatype=self, x=x, uri=uri)
        else:
            return self.encodable_cls(datatype=self, uri=uri)

    @ensure_initialized
    def encode_data(self, item, info: t.Optional[t.Dict] = None):
        """
        Encode item in bytes.

        :param item: Item to encode.
        :param info: Optional information.
        """
        info = info or {}
        data = self.encoder(item, info)
        data = self.bytes_encoding_after_encode(data)
        return data

    @ensure_initialized
    def decode_data(self, item, info: t.Optional[t.Dict] = None):
        """
        Decode item.

        :param item: Item to decode.
        :param info: Optional information.
        """
        info = info or {}
        item = self.bytes_encoding_before_decode(item)
        return self.decoder(item, info=info)

    def bytes_encoding_after_encode(self, data):
        """
        Convert data to bytes format.

        :param data: Data to encode.
        """
        if self.bytes_encoding == BytesEncoding.BASE64:
            return to_base64(data)
        return data

    def bytes_encoding_before_decode(self, data):
        """
        Perform bytes encoding before decoding.

        :param data: Data to decode.
        """
        if self.bytes_encoding == BytesEncoding.BASE64:
            return from_base64(data)
        return data


def encode_torch_state_dict(module, info):
    """
    Encode torch state dictionary.

    :param module: Module.
    :param info: Information.
    """
    import torch

    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)

    return buffer.getvalue()


class DecodeTorchStateDict:
    """
    Torch state dictionary decoder.

    :param cls: Torch state cls
    """

    def __init__(self, cls):
        self.cls = cls

    def __call__(self, b: bytes, info: t.Dict):
        import torch

        buffer = io.BytesIO(b)
        module = self.cls(**info)
        module.load_state_dict(torch.load(buffer))
        return module


def build_torch_state_serializer(module, info):
    """
    Datatype for serializing torch state dict.

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
    """
    Find descendants.
    """
    descendants = cls.__subclasses__()
    for subclass in descendants:
        descendants.extend(_find_descendants(subclass))
    return descendants


@dc.dataclass(kw_only=True)
class _BaseEncodable(Leaf):
    """
        Data variable wrapping encode-able item. Encoding is controlled by the referred
    -    to ``Encoder`` instance.

        :param encoder: Instance of ``Encoder`` controlling encoding.
        :param x: Wrapped content.
        :param uri: URI of the content, if any.
    """

    file_id: t.Optional[str] = None
    datatype: DataType
    uri: t.Optional[str] = None
    sha1: t.Optional[str] = None

    def _deep_flat_encode(self, cache):
        """
        Deep flat encode.

        :param cache: Cache.
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
        """
        Get ID.

        """
        assert self.file_id is not None
        return f'_{self.leaf_type}/{self.file_id}'

    def __post_init__(self):
        """
        Post-initialization hook.
        """
        if self.uri is not None and self.file_id is None:
            self.file_id = _construct_file_id_from_uri(self.uri)

        if self.uri and not re.match('^[a-z]{0,5}://', self.uri):
            self.uri = f'file://{self.uri}'

    @property
    def unique_id(self):
        """
        Get unique ID.
        """
        if self.file_id is not None:
            return self.file_id
        return str(id(self.x))

    @property
    def reference(self):
        """
        Get reference to datatype.
        """
        return self.datatype.reference

    def unpack(self, db):
        """
        Unpack the content of the `Encodable`.

        :param db: Datalayer instance.
        """
        return self.x

    @classmethod
    def get_encodable_cls(cls, name, default=None):
        """
        Get the subclass of the _BaseEncodable with the given name.
        All the registered subclasses must be subclasses of the _BaseEncodable.

        :param name: Name.
        :param default: Default class.
        :raises ValueError: If no subclass with the name is found.
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
        """
        Get object.

        :param db: Datalayer instance.
        :param r: Representation.
        """
        pass

    @classmethod
    @abstractmethod
    def decode(cls, r, db=None) -> '_BaseEncodable':
        """
        Decode method.

        :param r: Representation.
        :param db: Datalayer instance.
        """
        pass

    def get_hash(self, data):
        """
        Get hash.

        :param data: Data.
        """
        if isinstance(data, str):
            bytes_ = data.encode()
        elif isinstance(data, bytes):
            bytes_ = data
        else:
            raise ValueError(f'Unsupported data type: {type(data)}')

        return hashlib.sha1(bytes_).hexdigest()


class Empty:
    """
    Sentinel class.
    """

    def __repr__(self):
        """
        Get representation.

        """
        return '<EMPTY>'


@dc.dataclass
class Encodable(_BaseEncodable):
    """
    Encode non python datatypes to database.

    :param x: Encodable object
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
        if db is None:
            try:
                from superduperdb.components.datatype import serializers

                datatype = serializers[r['datatype']]
            except KeyError:
                raise Exception(
                    f'You specified a serializer which doesn\'t have a'
                    f' default value: {r["datatype"]}'
                )
        else:
            datatype = db.datatypes[r['datatype']]
        object = datatype.decode_data(r['bytes'], info=datatype.info)
        return object

    @override
    def encode(self, leaf_types_to_keep: t.Sequence = ()):
        """
        Encode `self.x` to dictionary format which could be serialized
        to a database.

        :param leaf_types_to_keep: leaf nodes to keep from encoding
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
        """
        Build ``Encodable`` with `r`.

        :param r: build from params 'r'
        """
        return cls(**r)

    def init(self, db):
        """
        Initilization method.

        :param db: Datalayer instance
        """
        pass

    @classmethod
    def decode(cls, r, db=None) -> 'Encodable':
        """
        Decode dictionary to `Encodable` instance.

        :param r: Object to decode
        :param db: Datalayer instance
        """
        object = cls._get_object(db, r['_content'])
        return cls(
            x=object,
            datatype=db.datatypes[r['_content']['datatype']],
            uri=r['_content']['uri'],
            file_id=r['_content'].get('file_id'),
        )


@dc.dataclass
class Native(_BaseEncodable):
    """
    Native data supported by underlying database.

    :param x: Encodable object
    """

    leaf_type: t.ClassVar[str] = 'native'
    x: t.Optional[t.Any] = None

    @classmethod
    def _get_object(cls, db, r):
        raise NotImplementedError

    @override
    def encode(self, leaf_types_to_keep: t.Sequence = ()):
        """
        Encode object.

        :param leaf_types_to_keep: leaf node to keep from
                                    encoding.
        """
        return self.x

    @classmethod
    def decode(cls, r, db=None):
        """
        Decode `r`.

        :param r: Object to decode
        :param db: Datalayer instance
        """
        return r


class _ArtifactSaveMixin:
    def save(self, artifact_store):
        r = artifact_store.save_artifact(self.encode()['_content'])
        self.x = None
        self.file_id = r['file_id']


@dc.dataclass
class Artifact(_BaseEncodable, _ArtifactSaveMixin):
    """
    Data to be saved on disk/ in the artifact-store.

    :param x: Artifact object
    :param artifact: If object is an artifact
    """

    leaf_type: t.ClassVar[str] = 'artifact'
    x: t.Any = Empty()
    artifact: bool = False

    def _encode(self):
        bytes_ = self.datatype.encode_data(self.x)
        sha1 = self.get_hash(bytes_)
        return bytes_, sha1

    def init(self, db):
        """
        Initialization to seed `x` with actual object from
        artifact store.

        :param db: A Datalayer instance
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
        """
        Encode `self.x` to dictionary format which later is saved
        in an artifact store.

        :param leaf_types_to_keep: Leaf nodes to exclude
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
        """
        Unpack the content of the `Encodable`

        :param db: `Datalayer` instance to assist with
        """
        self.init(db=db)
        return self.x

    def save(self, artifact_store):
        """
        Save the encoded data into an artifact store.

        :param artifact_store: Artifact store for storing
                                encoded object.
        """
        r = artifact_store.save_artifact(self.encode()['_content'])
        self.x = None
        self.file_id = r['file_id']

    @classmethod
    def decode(cls, r, db=None) -> 'Artifact':
        """
        Decode dictionary into instance of `Artifact`.

        :param r: Object to decode
        :param db: Datalayer instance
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
    """
    Data to be saved on disk/ in the artifact-store
    and loaded only when needed.

    :param artifact: If object is an artifact
    """

    leaf_type: t.ClassVar[str] = 'lazy_artifact'
    artifact: bool = False

    def __post_init__(self):
        if self.datatype.bytes_encoding == BytesEncoding.BASE64:
            raise ArtifactSavingError('BASE64 not supported on disk!')

    @override
    def encode(self, leaf_types_to_keep: t.Sequence = ()):
        """
        Encode `x` in dictionary format for artifact store.

        :param leaf_types_to_keep: Leaf nodes to exclude
                                   from encoding.
        """
        return super().encode(leaf_types_to_keep)

    def unpack(self, db):
        """
        Unpack the content of the `Encodable`.

        :param db: `Datalayer` instance to assist with
        """
        self.init(db=db)
        return self.x

    def save(self, artifact_store):
        """
        Save encoded data into artifact store.

        :param artifact_store: Artifact store for saving
                               encoded object.
        """
        r = artifact_store.save_artifact(self.encode()['_content'])
        self.x = None
        self.file_id = r['file_id']

    @classmethod
    def decode(cls, r, db=None) -> 'LazyArtifact':
        """
        Decode data into `LazyArtifact` instance.

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
    """
    Data to be saved on disk and passed
    as a file reference.

    :param x: File object
    """

    leaf_type: t.ClassVar[str] = 'file'
    x: t.Any = Empty()

    def init(self, db):
        """
        Initialization to laod `x` with actual file from
        artifact store.

        :param db: A Datalayer instance
        """
        if isinstance(self.x, Empty):
            file = db.artifact_store._load_file(self.file_id)
            self.x = file

    def unpack(self, db):
        """
        Unpack and get original data.

        :param db: Datalayer instance.
        """
        self.init(db)
        return self.x

    @classmethod
    def _get_object(cls, db, r):
        return r['x']

    @override
    def encode(self, leaf_types_to_keep: t.Sequence = ()):
        """
        Encode `x` to dictionary which is saved to
        artifact store later.

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
        """
        Decode data to `File` instance.

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
    """
    Class is used to load a file only when needed.
    """

    leaf_type: t.ClassVar[str] = 'lazy_file'

    @classmethod
    def decode(cls, r, db=None) -> 'LazyFile':
        """
        Decode dictionary to `LazyFile` instance.

        :param r: object to decode
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
serializers = {
    'pickle': pickle_serializer,
    'dill': dill_serializer,
    'torch': torch_serializer,
    'file': file_serializer,
    'pickle_lazy': pickle_lazy,
    'dill_lazy': dill_lazy,
    'file_lazy': file_lazy,
}
