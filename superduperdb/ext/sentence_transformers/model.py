import dataclasses as dc
import typing as t

from overrides import override
from sentence_transformers import SentenceTransformer as _SentenceTransformer

from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.components.datatype import DataType, dill_serializer
from superduperdb.components.model import _DeviceManaged, _Predictor


@dc.dataclass(kw_only=True)
class SentenceTransformer(_Predictor, _DeviceManaged):
    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, 'DataType']]] = (
        ('object', dill_serializer),
    )

    object: t.Optional[_SentenceTransformer] = None
    model: t.Optional[str] = None
    device: str = 'cpu'
    preprocess: t.Optional[t.Callable] = None

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

    @override
    def predict_one(self, X) -> int:
        if self.preprocess is not None:
            X = self.preprocess(X)

        assert self.object is not None
        return self.object.encode(X, self.predict_kwargs)

    @override
    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        if self.preprocess is not None:
            dataset = list(map(self.preprocess, dataset))  # type: ignore[arg-type]
        return self.object.encode(  # type: ignore[union-attr]
            dataset,
            **self.predict_kwargs,
        )
