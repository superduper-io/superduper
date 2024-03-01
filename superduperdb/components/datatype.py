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
from superduperdb.backends.base.artifact import (
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
    return pickle.dumps(object)


def pickle_decode(b: bytes, info: t.Optional[t.Dict] = None) -> t.Any:
    return pickle.loads(b)


def dill_encode(object: t.Any, info: t.Optional[t.Dict] = None) -> bytes:
    return dill.dumps(object, recurse=True)


def dill_decode(b: bytes, info: t.Optional[t.Dict] = None) -> t.Any:
    return dill.loads(b)


def file_check(path: t.Any, info: t.Optional[t.Dict] = None) -> str:
    if not (isinstance(path, str) and os.path.exists(path)):
        raise ValueError(f"Path '{path}' does not exist")
    return path


def torch_encode(object: t.Any, info: t.Optional[t.Dict] = None) -> bytes:
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
    import torch

    return torch.load(io.BytesIO(b))


def to_base64(bytes):
    return base64.b64encode(bytes).decode('utf-8')


def from_base64(encoded):
    return base64.b64decode(encoded)


@public_api(stability='stable')
@dc.dataclass(kw_only=True)
class DataType(Component):
    """
    {component_parameters}
    :param identifier: Unique identifier
    :param decoder: callable converting a ``bytes`` string to a ``Encodable``
                    of this ``Encoder``
    :param encoder: Callable converting an ``Encodable`` of this ``Encoder``
                    to ``bytes``
    :param shape: Shape of the data
    :param load_hybrid: Whether to load the data from the URI or return the URI in
                        `CFG.hybrid` mode
    """

    __doc__ = __doc__.format(component_parameters=Component.__doc__)

    type_id: t.ClassVar[str] = 'datatype'
    encoder: t.Callable = dill_encode
    decoder: t.Callable = dill_decode
    info: t.Optional[t.Dict] = None
    shape: t.Optional[t.Sequence] = None
    directory: t.Optional[str] = None
    encodable: str = 'encodable'
    bytes_encoding: str = CFG.bytes_encoding

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)
        self.encodable_cls = _ENCODABLES[self.encodable]
        self._takes_x = 'x' in inspect.signature(self.encodable_cls.__init__).parameters

    def __call__(
        self, x: t.Optional[t.Any] = None, uri: t.Optional[str] = None
    ) -> '_BaseEncodable':
        if self._takes_x:
            return self.encodable_cls(datatype=self, x=x, uri=uri)
        else:
            return self.encodable_cls(datatype=self, uri=uri)

    @ensure_initialized
    def encode_data(self, item, info: t.Optional[t.Dict] = None):
        info = info or {}
        try:
            return self.encoder(item, info)
        except:
            raise

    @ensure_initialized
    def decode_data(self, item, info: t.Optional[t.Dict] = None):
        info = info or {}
        return self.decoder(item, info=info)


def encode_torch_state_dict(module, info):
    import torch

    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)

    return buffer.getvalue()


class DecodeTorchStateDict:
    def __init__(self, cls):
        self.cls = cls

    def __call__(self, b: bytes, info: t.Dict):
        import torch

        buffer = io.BytesIO(b)
        module = self.cls(**info)
        module.load_state_dict(torch.load(buffer))
        return module


def build_torch_state_serializer(module, info):
    return DataType(
        identifier=module.__name__,
        info=info,
        encoder=encode_torch_state_dict,
        decoder=DecodeTorchStateDict(module),
    )


def _find_descendants(cls):
    descendants = cls.__subclasses__()
    for subclass in descendants:
        descendants.extend(_find_descendants(subclass))
    return descendants


@dc.dataclass(kw_only=True)
class _BaseEncodable(Leaf):

    """
    Data variable wrapping encode-able item. Encoding is controlled by the referred
    to ``Encoder`` instance.

    :param encoder: Instance of ``Encoder`` controlling encoding
    :param x: Wrapped content
    :param uri: URI of the content, if any
    """

    file_id: t.Optional[str] = None
    datatype: DataType
    uri: t.Optional[str] = None

    def __post_init__(self):
        if self.uri is not None and self.file_id is None:
            self.file_id = _construct_file_id_from_uri(self.uri)

        if self.uri and not re.match('^[a-z]{0,5}://', self.uri):
            self.uri = f'file://{self.uri}'

    @property
    def unique_id(self):
        return str(id(self.x))

    @property
    def artifact(self):
        return self.datatype.artifact

    @property
    def reference(self):
        return self.datatype.reference

    def unpack(self, db):
        """
        Unpack the content of the `Encodable`

        :param db: `Datalayer` instance to assist with
        """
        return self.x

    @classmethod
    def get_encodable_cls(cls, name, default=None):
        """
        Get the subclass of the _BaseEncodable with the given name.
        All the registered subclasses must be subclasses of the _BaseEncodable.
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
        pass

    @classmethod
    @abstractmethod
    def decode(cls, r, db=None) -> '_BaseEncodable':
        pass


@dc.dataclass
class Encodable(_BaseEncodable):
    x: t.Optional[t.Any] = None
    leaf_type: t.ClassVar[str] = 'encodable'

    def _encode(self):
        bytes_ = self.datatype.encode_data(self.x)
        sha1 = str(hashlib.sha1(bytes_).hexdigest())
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
        bytes_, sha1 = self._encode()
        if self.datatype.bytes_encoding == BytesEncoding.BASE64:
            bytes_ = to_base64(bytes_)
        return {
            '_content': {
                'bytes': bytes_,
                'datatype': self.datatype.identifier,
                'leaf_type': self.leaf_type,
                'sha1': sha1,
                'uri': self.uri,
                'file_id': sha1 if self.file_id is None else self.file_id,
            }
        }

    def init(self, db):
        pass

    @classmethod
    def decode(cls, r, db=None) -> 'Encodable':
        object = cls._get_object(db, r['_content'])
        return cls(
            x=object,
            datatype=db.datatypes[r['_content']['datatype']],
            uri=r['_content']['uri'],
            file_id=r['_content'].get('file_id'),
        )


class Empty:
    def __repr__(self):
        return '<EMPTY>'


@dc.dataclass
class Native(_BaseEncodable):
    """
    Native data supported by underlying database
    """

    leaf_type: t.ClassVar[str] = 'native'
    x: t.Optional[t.Any] = None

    @classmethod
    def _get_object(cls, db, r):
        raise NotImplementedError

    @override
    def encode(self, leaf_types_to_keep: t.Sequence = ()):
        return self.x

    @classmethod
    def decode(cls, r, db=None):
        return r


@dc.dataclass
class Artifact(_BaseEncodable):
    """
    Data to be saved on disk/ in the artifact-store
    """

    leaf_type: t.ClassVar[str] = 'artifact'
    x: t.Optional[t.Any] = None
    artifact: bool = False

    def __post_init__(self):
        if self.datatype.bytes_encoding == BytesEncoding.BASE64:
            raise ArtifactSavingError('BASE64 not supported on disk!')

    def _encode(self):
        bytes_ = self.datatype.encode_data(self.x)
        sha1 = str(hashlib.sha1(bytes_).hexdigest())
        return bytes_, sha1

    @override
    def encode(self, leaf_types_to_keep: t.Sequence = ()):
        bytes_, sha1 = self._encode()
        if self.datatype.bytes_encoding == BytesEncoding.BASE64:
            bytes_ = to_base64(bytes_)
        return {
            '_content': {
                'bytes': bytes_,
                'datatype': self.datatype.identifier,
                'leaf_type': self.leaf_type,
                'sha1': sha1,
                'uri': self.uri,
                'file_id': sha1 if self.file_id is None else self.file_id,
            }
        }

    def init(self, db):
        pass

    @classmethod
    def _get_object(cls, db, file_id, datatype, uri):
        return db.artifact_store.load_artifact(
            {
                'file_id': file_id,
                'datatype': datatype,
                'uri': uri,
            }
        )

    def unpack(self, db):
        """
        Unpack the content of the `Encodable`

        :param db: `Datalayer` instance to assist with
        """
        self.init(db=db)
        return self.x

    def save(self, artifact_store):
        r = artifact_store.save_artifact(self.encode()['_content'])
        self.x = None
        self.file_id = r['file_id']

    @classmethod
    def decode(cls, r, db=None) -> 'Artifact':
        r = r['_content']
        x = cls._get_object(
            db, file_id=r['file_id'], datatype=r['datatype'], uri=r['uri']
        )
        return cls(
            x=x,
            datatype=db.datatypes[r['datatype']],
            uri=r['uri'],
            file_id=r.get('file_id'),
        )


@dc.dataclass
class LazyArtifact(Artifact):
    """
    Data to be saved on disk/ in the artifact-store
    and loaded only when needed
    """

    leaf_type: t.ClassVar[str] = 'lazy_artifact'
    artifact: bool = False

    def __post_init__(self):
        if self.datatype.bytes_encoding == BytesEncoding.BASE64:
            raise ArtifactSavingError('BASE64 not supported on disk!')

    def init(self, db):
        if isinstance(self.x, Empty):
            self.x = self._get_object(
                db,
                file_id=self.file_id,
                datatype=self.datatype.identifier,
                uri=self.uri,
            )

    @override
    def encode(self, leaf_types_to_keep: t.Sequence = ()):
        return super().encode(leaf_types_to_keep)

    def unpack(self, db):
        """
        Unpack the content of the `Encodable`

        :param db: `Datalayer` instance to assist with
        """
        self.init(db=db)
        return self.x

    def save(self, artifact_store):
        r = artifact_store.save_artifact(self.encode()['_content'])
        self.x = None
        self.file_id = r['file_id']

    @classmethod
    def decode(cls, r, db=None) -> 'LazyArtifact':
        return cls(
            x=Empty(),  # this is to enable lazy loading
            datatype=db.datatypes[r['_content']['datatype']],
            uri=r['_content']['uri'],
            file_id=r['_content'].get('file_id'),
        )


@dc.dataclass
class File(_BaseEncodable):
    """
    Data to be saved on disk and passed
    as a file reference
    """

    leaf_type: t.ClassVar[str] = 'file'

    def __post_init__(self):
        if self.file_id is None:
            self.file_id = random_sha1()

    def init(self, db):
        file = db.artifact_store._load_file(self.file_id)
        self.uri = f'file://{file}'

    def unpack(self, db):
        self.init(db)
        uri = self.uri.split('file://')[-1]
        assert not re.match('^[a-z]{0,5}://', uri)
        return uri

    @classmethod
    def _get_object(cls, db, r):
        return r['x']

    @override
    def encode(self, leaf_types_to_keep: t.Sequence = ()):
        dc.asdict(self)
        return {
            '_content': {
                'datatype': self.datatype.identifier,
                'leaf_type': self.leaf_type,
                'uri': self.uri,
                'file_id': self.file_id,
            }
        }

    @classmethod
    def decode(cls, r, db=None) -> 'File':
        return cls(
            datatype=db.datatypes[r['_content']['datatype']],
            uri=r['_content']['uri'],
            file_id=r['_content']['file_id'],
        )


Encoder = DataType


_ENCODABLES = {
    'encodable': Encodable,
    'artifact': Artifact,
    'file': File,
    'native': Native,
    'lazy_artifact': LazyArtifact,
}


pickle_serializer = DataType(
    'pickle',
    encoder=pickle_encode,
    decoder=pickle_decode,
    encodable='artifact',
)
pickle_lazy = DataType(
    'pickle',
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
    'dill',
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
serializers = {
    'pickle': pickle_serializer,
    'dill': dill_serializer,
    'torch': torch_serializer,
    'file': file_serializer,
}
