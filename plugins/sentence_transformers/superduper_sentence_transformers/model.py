import typing as t

from sentence_transformers import SentenceTransformer as _SentenceTransformer
from superduper.base.query_dataset import QueryDataset
from superduper.components.component import ensure_setup
from superduper.components.model import Model, _DeviceManaged

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

    Example:
    -------
    >>> from superduper import vector
    >>> from superduper_sentence_transformers import SentenceTransformer
    >>> import sentence_transformers
    >>> model = SentenceTransformer(
    >>>     identifier="embedding",
    >>>     object=sentence_transformers.SentenceTransformer("BAAI/bge-small-en"),
    >>>     datatype=vector(shape=(1024,)),
    >>>     postprocess=lambda x: x.tolist(),
    >>>     predict_kwargs={"show_progress_bar": True},
    >>> )
    >>> model.predict("What is superduper")

    """

    object: t.Optional[_SentenceTransformer] = None
    model: t.Optional[str] = None
    device: str = 'cpu'
    preprocess: t.Union[None, t.Callable] = None
    postprocess: t.Union[None, t.Callable] = None

    def postinit(self):
        """Post-initialization method."""
        if self.model is None:
            self.model = self.identifier

        self._default_model = False
        if self.object is None:
            self.object = _SentenceTransformer(self.model, device=self.device)
            self._default_model = True
        return super().postinit()

    def setup(self):
        """Initialize the model."""
        super().setup()
        self.to(self.device)

    def to(self, device):
        """Move the model to a device.

        :param device: The device to move to, e.g. 'cpu' or 'cuda'.
        """
        self.object = self.object.to(device)
        self.object._target_device = device

    @ensure_setup
    def predict(self, text):
        """Predict on a single input.

        :param text: The input to predict on.
        """
        if self.preprocess is not None:
            text = self.preprocess(text)

        assert self.object is not None
        result = self.object.encode(text, **self.predict_kwargs)
        if self.postprocess is not None:
            result = self.postprocess(result)
        return result

    @t.no_type_check
    @ensure_setup
    def predict_batches(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        """Predict on a dataset.

        :param dataset: The dataset to predict on.
        """
        if self.preprocess is not None:
            dataset = list(map(self.preprocess, dataset))
        assert self.object is not None

        results = self.object.encode(dataset, **self.predict_kwargs)
        if self.postprocess is not None:
            results = self.postprocess(results)
        return results
