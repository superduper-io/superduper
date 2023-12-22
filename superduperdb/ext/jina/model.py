import dataclasses as dc
import typing as t

import tqdm

from superduperdb.backends.ibis.data_backend import IbisDataBackend
from superduperdb.components.model import APIModel
from superduperdb.components.vector_index import sqlvector, vector
from superduperdb.ext.jina.client import JinaAPIClient


@dc.dataclass(kw_only=True)
class Jina(APIModel):
    """Cohere predictor"""

    api_key: t.Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.identifier = self.identifier or self.model
        self.client = JinaAPIClient(model_name=self.identifier, api_key=self.api_key)


@dc.dataclass(kw_only=True)
class JinaEmbedding(Jina):
    """Jina embedding predictor

    :param shape: The shape of the embedding as ``tuple``.
        If not provided, it will be obtained by sending a simple query to the API
    """

    shape: t.Optional[t.Sequence[int]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.shape is None:
            self.shape = (len(self.client.encode_batch(['shape'])[0]),)

    def pre_create(self, db):
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend):
            if self.encoder is None:
                self.encoder = sqlvector(self.shape)
        elif self.encoder is None:
            self.encoder = vector(self.shape)

    def _predict_one(self, X: str, **kwargs):
        return self.client.encode_batch([X])[0]

    async def _apredict_one(self, X: str, **kwargs):
        embeddings = await self.client.aencode_batch([X])
        return embeddings[0]

    def _predict_a_batch(self, texts: t.List[str], **kwargs):
        return self.client.encode_batch(texts)

    async def _apredict_a_batch(self, texts: t.List[str], **kwargs):
        return await self.client.aencode_batch(texts)

    def _predict(self, X, one=False, **kwargs):
        if isinstance(X, str):
            return self._predict_one(X)
        out = []
        batch_size = kwargs.pop('batch_size', 100)
        for i in tqdm.tqdm(range(0, len(X), batch_size)):
            out.extend(self._predict_a_batch(X[i : i + batch_size], **kwargs))
        return out

    async def _apredict(self, X, one=False, **kwargs):
        if isinstance(X, str):
            return await self._apredict_one(X)
        out = []
        batch_size = kwargs.pop('batch_size', 100)
        for i in range(0, len(X), batch_size):
            out.extend(await self._apredict_a_batch(X[i : i + batch_size], **kwargs))
        return out
