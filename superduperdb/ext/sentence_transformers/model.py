import typing as t

from sentence_transformers import SentenceTransformer as _SentenceTransformer

from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.base.enums import DBType
from superduperdb.components.component import ensure_initialized
from superduperdb.components.datatype import DataType, dill_lazy
from superduperdb.components.model import Model, Signature, _DeviceManaged

DEFAULT_PREDICT_KWARGS = {
    'show_progress_bar': True,
}


class SentenceTransformer(Model, _DeviceManaged):
    """A model for sentence embeddings using `sentence-transformers`.

    :param object: The SentenceTransformer object to use.
    :param model: The model name, e.g. 'all-MiniLM-L6-v2'.
    :param device: The device to use, e.g. 'cpu' or 'cuda'.
    :param preprocess: The preprocessing function to apply to the input.
    :param postprocess: The postprocessing function to apply to the output.
    :param signature: The signature of the model.
    :param preferred_devices: A list of devices to prefer, in that order.
    """

    _artifacts: t.ClassVar[t.Sequence[t.Tuple[str, 'DataType']]] = (
        ('object', dill_lazy),
    )

    object: t.Optional[_SentenceTransformer] = None
    model: t.Optional[str] = None
    device: str = 'cpu'
    preprocess: t.Union[None, t.Callable] = None
    postprocess: t.Union[None, t.Callable] = None
    signature: Signature = 'singleton'

    def __post_init__(self, db, artifacts):
        super().__post_init__(db, artifacts)

        if self.model is None:
            self.model = self.identifier

        self._default_model = False
        if self.object is None:
            self.object = _SentenceTransformer(self.model, device=self.device)
            self._default_model = True

        if self.datatype is None:
            sample = self.predict('Test')
            self.shape = (len(sample),)

    def dict(self):
        """Serialize as a dictionary."""
        r = super().dict()
        if self._default_model:
            del r['object']
        return r

    def init(self, db=None):
        """Initialize the model."""
        super().init(db=db)
        self.to(self.device)

    def to(self, device):
        """Move the model to a device.

        :param device: The device to move to, e.g. 'cpu' or 'cuda'.
        """
        self.object = self.object.to(device)
        self.object._target_device = device

    @ensure_initialized
    def predict(self, X, *args, **kwargs):
        """Predict on a single input.

        :param X: The input to predict on.
        :param args: Additional positional arguments, which are passed to the model.
        :param kwargs: Additional keyword arguments, which are passed to the model.
        """
        if self.preprocess is not None:
            X = self.preprocess(X)

        assert self.object is not None
        result = self.object.encode(X, *args, **{**self.predict_kwargs, **kwargs})
        if self.postprocess is not None:
            result = self.postprocess(result)
        return result

    @ensure_initialized
    def predict_batches(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        """Predict on a dataset.

        :param dataset: The dataset to predict on.
        """
        if self.preprocess is not None:
            dataset = list(map(self.preprocess, dataset))  # type: ignore[arg-type]
        assert self.object is not None
        results = self.object.encode(dataset, **self.predict_kwargs)
        if self.postprocess is not None:
            results = self.postprocess(results)
        return results

    def pre_create(self, db):
        """Pre creates the model.

        If the datatype is not set and the datalayer is an IbisDataBackend,
        the datatype is set to ``sqlvector`` or ``vector``.

        :param db: The datalayer instance.
        """
        super().pre_create(db)
        if self.datatype is not None:
            return

        from superduperdb.components.vector_index import sqlvector, vector

        if db.databackend.db_type == DBType.SQL:
            self.datatype = sqlvector(shape=self.shape)
        else:
            self.datatype = vector(shape=self.shape)
