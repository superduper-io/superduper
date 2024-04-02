import dataclasses as dc
import typing as t

from superduperdb.misc.annotations import requires_packages

from sentence_transformers import SentenceTransformer as _SentenceTransformer

from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.components.datatype import DataType, dill_serializer
from superduperdb.components.model import Model, Signature, _DeviceManaged


@dc.dataclass(kw_only=True)
class SentenceTransformer(Model, _DeviceManaged):
    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, 'DataType']]] = (
        ('object', dill_serializer),
    )

    object: t.Optional[_SentenceTransformer] = None
    model: t.Optional[str] = None
    device: str = 'cpu'
    preprocess: t.Optional[t.Callable] = None
    postprocess: t.Optional[t.Callable] = None
    signature: Signature = 'singleton'

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)

        if self.model is None:
            self.model = self.identifier

        if self.object is None:
            self.object = _SentenceTransformer(self.identifier, device=self.device)

        self.to(self.device)

    def to(self, device):
        self.object = self.object.to(device)
        self.object._target_device = device

    def predict_one(self, X, *args, **kwargs):
        if self.preprocess is not None:
            X = self.preprocess(X)

        assert self.object is not None
        result = self.object.encode(X, *args, **{**self.predict_kwargs, **kwargs})
        if self.postprocess is not None:
            result = self.postprocess(result)
        return result

    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        if self.preprocess is not None:
            dataset = list(map(self.preprocess, dataset))  # type: ignore[arg-type]
        assert self.object is not None
        results = self.object.encode(dataset, **self.predict_kwargs)
        if self.postprocess is not None:
            results = self.postprocess(results)
        return results
