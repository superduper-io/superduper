import typing as t

import tqdm

from superduperdb.backends.ibis.data_backend import IbisDataBackend
from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.components.model import APIBaseModel
from superduperdb.components.vector_index import sqlvector, vector
from superduperdb.ext.jina.client import JinaAPIClient


class Jina(APIBaseModel):
    """Cohere predictor.

    :param api_key: The API key to use for the predicto
    """

    api_key: t.Optional[str] = None

    def __post_init__(self, db, artifacts):
        super().__post_init__(db, artifacts)
        self.identifier = self.identifier or self.model
        self.client = JinaAPIClient(model_name=self.identifier, api_key=self.api_key)


class JinaEmbedding(Jina):
    """Jina embedding predictor.

    :param batch_size: The batch size to use for the predictor.
    :param shape: The shape of the embedding as ``tuple``.
        If not provided, it will be obtained by sending a simple query to the API
    """

    batch_size: int = 100
    shape: t.Optional[t.Sequence[int]] = None
    signature: str = 'singleton'

    def __post_init__(self, db, artifacts):
        super().__post_init__(db, artifacts)
        if self.shape is None:
            self.shape = (len(self.client.encode_batch(['shape'])[0]),)

    def pre_create(self, db):
        """Pre create method for the model.

        If the datalayer is Ibis, the datatype will be set to the appropriate
        SQL datatype.

        :param db: The datalayer to use for the model.
        """
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend):
            if self.datatype is None:
                self.datatype = sqlvector(self.shape)
        elif self.datatype is None:
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
