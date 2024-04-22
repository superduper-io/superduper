import dataclasses as dc
import typing as t

from sentence_transformers import SentenceTransformer as _SentenceTransformer

from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.base.code import Code
from superduperdb.components.component import ensure_initialized
from superduperdb.components.datatype import DataType, dill_serializer
from superduperdb.components.model import Model, Signature, _DeviceManaged

DEFAULT_PREDICT_KWARGS = {
    'show_progress_bar': True,
}


@dc.dataclass(kw_only=True)
class SentenceTransformer(Model, _DeviceManaged):
    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, 'DataType']]] = (
        ('object', dill_serializer),
    )

    object: t.Optional[_SentenceTransformer] = None
    model: t.Optional[str] = None
    device: str = 'cpu'
    preprocess: t.Union[None, t.Callable, Code] = None
    postprocess: t.Union[None, t.Callable, Code] = None
    signature: Signature = 'singleton'

    ui_schema: t.ClassVar[t.List[t.Dict]] = [
        {'name': 'model', 'type': 'str', 'default': 'all-MiniLM-L6-v2'},
        {'name': 'device', 'type': 'str', 'default': 'cpu', 'choices': ['cpu', 'cuda']},
        {'name': 'predict_kwargs', 'type': 'json', 'default': DEFAULT_PREDICT_KWARGS},
        {'name': 'postprocess', 'type': 'code', 'default': Code.default},
    ]

    @classmethod
    def handle_integration(cls, kwargs):
        if isinstance(kwargs.get('preprocess'), str):
            kwargs['preprocess'] = Code(kwargs['preprocess'])
        if isinstance(kwargs.get('postprocess'), str):
            kwargs['postprocess'] = Code(kwargs['postprocess'])
        return kwargs

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)

        if self.model is None:
            self.model = self.identifier

        if self.object is None:
            self.object = _SentenceTransformer(self.model, device=self.device)

        self.to(self.device)

    def to(self, device):
        self.object = self.object.to(device)
        self.object._target_device = device

    @ensure_initialized
    def predict_one(self, X, *args, **kwargs):
        if self.preprocess is not None:
            X = self.preprocess(X)

        assert self.object is not None
        result = self.object.encode(X, *args, **{**self.predict_kwargs, **kwargs})
        if self.postprocess is not None:
            result = self.postprocess(result)
        return result

    @ensure_initialized
    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        if self.preprocess is not None:
            dataset = list(map(self.preprocess, dataset))  # type: ignore[arg-type]
        assert self.object is not None
        results = self.object.encode(dataset, **self.predict_kwargs)
        if self.postprocess is not None:
            results = self.postprocess(results)  # type: ignore[operator]
        return results
