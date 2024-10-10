import typing as t

import tqdm
from superduper.backends.query_dataset import QueryDataset
from superduper.components.model import APIBaseModel
from superduper.components.vector_index import vector

from superduper_jina.client import JinaAPIClient


class Jina(APIBaseModel):
    """Cohere predictor.

    :param api_key: The API key to use for the predicto
    """

    api_key: t.Optional[str] = None

    def __post_init__(self, db, artifacts, example):
        super().__post_init__(db, artifacts, example=example)
        self.identifier = self.identifier or self.model
        self.client = JinaAPIClient(model_name=self.identifier, api_key=self.api_key)


class JinaEmbedding(Jina):
    """Jina embedding predictor.

    :param batch_size: The batch size to use for the predictor.
    :param shape: The shape of the embedding as ``tuple``.
        If not provided, it will be obtained by sending a simple query to the API

    Example:
    -------
    >>> from superduper_jina.model import JinaEmbedding
    >>> model = JinaEmbedding(identifier='jina-embeddings-v2-base-en')
    >>> model.predict('Hello world')

    """

    batch_size: int = 100
    shape: t.Optional[t.Sequence[int]] = None
    signature: str = 'singleton'

    def __post_init__(self, db, artifacts, example):
        super().__post_init__(db, artifacts, example)
        if self.shape is None:
            self.shape = (len(self.client.encode_batch(['shape'])[0]),)

    def _pre_create(self, db):
        """Pre create method for the model.

        If the datalayer is Ibis, the datatype will be set to the appropriate
        SQL datatype.

        :param db: The datalayer to use for the model.
        """
        if self.datatype is None:
            self.datatype = vector(shape=self.shape)

    def predict(self, X: str):
        """Predict the embedding of a single text.

        :param X: The text to predict the embedding of.
        """
        return self.client.encode_batch([X])[0]

    def _predict_a_batch(self, texts: t.List[str]):
        return self.client.encode_batch(texts)

    def predict_batches(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        """Predict the embeddings of a dataset.

        :param dataset: The dataset to predict the embeddings of.
        """
        out = []
        for i in tqdm.tqdm(range(0, len(dataset), self.batch_size)):
            batch = [
                dataset[i] for i in range(i, min(i + self.batch_size, len(dataset)))
            ]
            out.extend(self._predict_a_batch(batch))
        return out
