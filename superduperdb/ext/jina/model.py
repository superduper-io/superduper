import dataclasses as dc
import typing as t

import tqdm

from superduperdb.backends.ibis.data_backend import IbisDataBackend
from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.components.model import APIModel
from superduperdb.components.vector_index import sqlvector, vector
from superduperdb.ext.jina.client import JinaAPIClient


@dc.dataclass(kw_only=True)
class Jina(APIModel):
    """Cohere predictor"""

    api_key: t.Optional[str] = None

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)
        self.identifier = self.identifier or self.model
        self.client = JinaAPIClient(model_name=self.identifier, api_key=self.api_key)


@dc.dataclass(kw_only=True)
class JinaEmbedding(Jina):
    """Jina embedding predictor

    :param shape: The shape of the embedding as ``tuple``.
        If not provided, it will be obtained by sending a simple query to the API
    """

    batch_size: int = 100
    signature: t.ClassVar[str] = 'singleton'

    shape: t.Optional[t.Sequence[int]] = None

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)
        if self.shape is None:
            self.shape = (len(self.client.encode_batch(['shape'])[0]),)

    def pre_create(self, db):
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend):
            if self.datatype is None:
                self.datatype = sqlvector(self.shape)
        elif self.datatype is None:
            self.datatype = vector(self.shape)

    def predict_one(self, X: str):
        return self.client.encode_batch([X])[0]

    def _predict_a_batch(self, texts: t.List[str]):
        return self.client.encode_batch(texts)

    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        out = []
        for i in tqdm.tqdm(range(0, len(dataset), self.batch_size)):
            batch = [
                dataset[i] for i in range(i, min(i + self.batch_size, len(dataset)))
            ]
            out.extend(self._predict_a_batch(batch))
        return out
