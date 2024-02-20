import base64
import dataclasses as dc
import hashlib
import io
import pickle
import typing as t

import dill

from superduperdb import CFG
from superduperdb.base.config import BytesEncoding
from superduperdb.base.leaf import Leaf
from superduperdb.components.component import Component
from superduperdb.misc.annotations import public_api

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
    artifact: bool = False
    reference: bool = False
    directory: t.Optional[str] = None

    def __call__(
        self, x: t.Optional[t.Any] = None, uri: t.Optional[str] = None
    ) -> 'Encodable':
        return Encodable(self, x=x, uri=uri)


pickle_serializer = DataType(
    'pickle', encoder=pickle_encode, decoder=pickle_decode, artifact=True
)
dill_serializer = DataType(
    'dill', encoder=dill_encode, decoder=dill_decode, artifact=True
)
torch_serializer = DataType(
    'torch', encoder=torch_encode, decoder=torch_decode, artifact=True
)
serializers = {
    'pickle': pickle_serializer,
    'dill': dill_serializer,
    'torch': torch_serializer,
}


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


@dc.dataclass
class Encodable(Leaf):
    """
    Data variable wrapping encode-able item. Encoding is controlled by the referred
    to ``Encoder`` instance.

    :param encoder: Instance of ``Encoder`` controlling encoding
    :param x: Wrapped content
    :param uri: URI of the content, if any
    """

    leaf_type: t.ClassVar[str] = 'encodable'
    datatype: DataType
    x: t.Optional[t.Any] = None
    uri: t.Optional[str] = None
    file_id: t.Optional[str] = None

    @property
    def unique_id(self):
        return str(id(self.x))

    @property
    def artifact(self):
        return self.datatype.artifact

    @property
    def reference(self):
        return self.datatype.reference

    def init(self, db):
        self.x = db.artifact_store.load_artifact(self.file_id)

    def unpack(self, db):
        """
        Unpack the content of the `Encodable`

        :param db: `Datalayer` instance to assist with
        """
        if self.x is None:
            self.init()
        return self.x

    def encode(
        self,
        bytes_encoding: t.Optional[BytesEncoding] = None,
        leaf_types_to_keep: t.Sequence = (),
    ) -> t.Union[t.Optional[str], t.Dict[str, t.Any]]:
        """
        :param bytes_encoding:
        """
        from superduperdb.backends.base.artifact import ArtifactSavingError

        def _encode(x):
            try:
                bytes_ = self.datatype.encoder(x)
            except Exception as e:
                raise ArtifactSavingError from e
            sha1 = str(hashlib.sha1(bytes_).hexdigest())
            if (
                CFG.bytes_encoding == BytesEncoding.BASE64
                or bytes_encoding == BytesEncoding.BASE64
            ):
                bytes_ = to_base64(bytes_)
            return bytes_, sha1

        if self.datatype.encoder is None:
            return self.x

        bytes_, sha1 = _encode(self.x)
        return {
            '_content': {
                'bytes': bytes_,
                'datatype': self.datatype.identifier,
                'leaf_type': 'encodable',
                'sha1': sha1,
                'uri': self.uri,
            }
        }

    @classmethod
    def decode(cls, r, db=None, reference: bool = False):
        # TODO tidy up this logic by creating different subclasses of datatype
        # Idea
        if 'bytes' in r['_content']:
            if db is None:
                try:
                    from superduperdb.components.datatype import serializers

                    datatype = serializers[r['_content']['datatype']]
                except KeyError:
                    raise Exception(
                        f'You specified a serializer which doesn\'t have a'
                        f' default value: {r["_content"]["datatype"]}'
                    )
            else:
                datatype = db.datatypes[r['_content']['datatype']]

            object = datatype.decoder(r['_content']['bytes'], info=datatype.info)
        else:
            datatype = db.datatypes[r['_content']['datatype']]
            if datatype.artifact and not datatype.reference and not reference:
                object = db.artifact_store.load_artifact(r['_content'])
            elif datatype.artifact and datatype.reference:
                return Encodable(x=None, datatype=datatype, uri=r['_content']['uri'])
            elif 'bytes' not in r['_content'] and reference:
                assert (
                    'uri' in r['_content']
                ), 'If load by reference, need a valid URI for data, found "None"'
                return Encodable(x=None, datatype=datatype, uri=r['_content']['uri'])
            else:
                raise Exception('Incorrect argument combination.')
        return Encodable(
            x=object,
            datatype=datatype,
            uri=r['_content']['uri'],
        )


Encoder = DataType
