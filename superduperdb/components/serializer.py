import dataclasses as dc
import io
import pickle
import typing as t

import dill

from superduperdb.base.artifact import Artifact
from superduperdb.components.component import Component
from superduperdb.misc.annotations import public_api

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer


@public_api(stability='beta')
@dc.dataclass(kw_only=True)
class Serializer(Component):
    """
    A component carrying the information to apply a serializer to a
    model.
    {component_parameters}
    :param object: The serializer
    """

    __doc__ = __doc__.format(component_parameters=Component.__doc__)

    type_id: t.ClassVar[str] = 'serializer'

    encoder: t.Union[t.Callable, Artifact]
    decoder: t.Union[t.Callable, Artifact]
    constructor: t.Union[t.Callable, Artifact, str, None] = None

    info: t.Optional[t.Dict] = None

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.encoder, Artifact):
            self.encoder = Artifact(self.encoder)
        if not isinstance(self.decoder, Artifact):
            self.decoder = Artifact(self.decoder)
        if self.constructor and not isinstance(self.constructor, Artifact):
            self.constructor = Artifact(self.constructor)

    def pre_create(self, db: 'Datalayer'):
        super().pre_create(db)
        self.object = t.cast(t.Type, self.identifier)

    def encode(self, object: t.Any) -> bytes:
        assert isinstance(self.encoder, Artifact)
        return self.encoder.artifact(object, info=self.info)

    def decode(self, object: t.Any) -> bytes:
        assert isinstance(self.decoder, Artifact)
        return self.decoder.artifact(object, info=self.info)

    def construct(self):
        assert isinstance(self.constructor, Artifact)
        return self.constructor.artifact(**self.info)


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


class Serializers:
    serializers: t.Dict[str, Serializer] = {}

    def __iter__(self):
        return iter(self.serializers)

    def add(self, name: str, serializer: Serializer):
        self.serializers[name] = serializer

    def __getitem__(self, serializer):
        return self.serializers[serializer]


serializers = Serializers()
serializers.add(
    'pickle', Serializer('pickle', encoder=pickle_encode, decoder=pickle_decode)
)
serializers.add('dill', Serializer('dill', encoder=dill_encode, decoder=dill_decode))
serializers.add(
    'torch', Serializer('torch', encoder=torch_encode, decoder=torch_decode)
)


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
    return Serializer(
        identifier=module.__name__,
        info=info,
        encoder=encode_torch_state_dict,
        decoder=DecodeTorchStateDict(module),
    )
