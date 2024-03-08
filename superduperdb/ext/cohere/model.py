import dataclasses as dc
import typing as t

import cohere
import tqdm
from cohere.error import CohereAPIError, CohereConnectionError

from superduperdb.backends.ibis.data_backend import IbisDataBackend
from superduperdb.backends.ibis.field_types import dtype
from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.base.datalayer import Datalayer
from superduperdb.components.model import APIModel
from superduperdb.components.vector_index import sqlvector, vector
from superduperdb.ext.utils import format_prompt, get_key
from superduperdb.misc.retry import Retry

retry = Retry(exception_types=(CohereAPIError, CohereConnectionError))

KEY_NAME = 'COHERE_API_KEY'


@dc.dataclass(kw_only=True)
class Cohere(APIModel):
    """Cohere predictor"""

    client_kwargs: t.Dict[str, t.Any] = dc.field(default_factory=dict)

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)
        self.identifier = self.identifier or self.model


@dc.dataclass(kw_only=True)
class CohereEmbed(Cohere):
    """Cohere embedding predictor

    :param shape: The shape as ``tuple`` of the embedding.
    """

    signature: t.ClassVar[str] = 'singleton'
    shapes: t.ClassVar[t.Dict] = {'embed-english-v2.0': (4096,)}
    shape: t.Optional[t.Sequence[int]] = None
    batch_size: int = 100

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)
        if self.shape is None:
            self.shape = self.shapes[self.identifier]

    def pre_create(self, db):
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend):
            if self.datatype is None:
                self.datatype = sqlvector(self.shape)
        elif self.datatype is None:
            self.datatype = vector(self.shape)

    @retry
    def predict_one(self, X: str):
        client = cohere.Client(get_key(KEY_NAME), **self.client_kwargs)
        e = client.embed(texts=[X], model=self.identifier, **self.predict_kwargs)
        return e.embeddings[0]

    @retry
    def _predict_a_batch(self, texts: t.List[str]):
        client = cohere.Client(get_key(KEY_NAME), **self.client_kwargs)
        out = client.embed(texts=texts, model=self.identifier, **self.predict_kwargs)
        return [r for r in out.embeddings]

    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        out = []
        for i in tqdm.tqdm(range(0, len(dataset), self.batch_size)):
            out.extend(
                self._predict_a_batch(
                    dataset[i : i + self.batch_size], **self.predict_kwargs
                )
            )
        return out


@dc.dataclass(kw_only=True)
class CohereGenerate(Cohere):
    """Cohere realistic text generator (chat predictor)

    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to use to seed the response.
    """

    signature: t.ClassVar[str] = '*args,**kwargs'
    takes_context: bool = True
    prompt: str = ''

    def pre_create(self, db: Datalayer) -> None:
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.datatype is None:
            self.datatype = dtype('str')

    @retry
    def predict_one(self, prompt: str, context: t.Optional[t.List[str]] = None):
        if context is not None:
            prompt = format_prompt(prompt, self.prompt, context=context)
        client = cohere.Client(get_key(KEY_NAME), **self.client_kwargs)
        resp = client.generate(
            prompt=prompt, model=self.identifier, **self.predict_kwargs
        )
        return resp.generations[0].text

    @retry
    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        return [self.predict_one(dataset[i]) for i in range(len(dataset))]
