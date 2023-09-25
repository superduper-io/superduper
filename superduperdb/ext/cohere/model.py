import dataclasses as dc
import typing as t

import cohere
import tqdm
from cohere.error import CohereAPIError, CohereConnectionError

from superduperdb.container.component import Component
from superduperdb.container.encoder import Encoder
from superduperdb.container.model import PredictMixin
from superduperdb.ext.utils import format_prompt, get_key
from superduperdb.ext.vector.encoder import vector
from superduperdb.misc.retry import Retry

retry = Retry(exception_types=(CohereAPIError, CohereConnectionError))

KEY_NAME = 'COHERE_API_KEY'


@dc.dataclass
class Cohere(Component, PredictMixin):
    """Cohere predictor

    :param model: The model to use, e.g. ``'base-light'``.
    :param identifier: The identifier to use, e.g. ``'my-model'``.
    :param version: The version to use, e.g. ``0`` (leave empty)
    :param takes_context: Whether the model takes context into account.
    :param encoder: The encoder identifier.
    :param client_kwargs: Keyword arguments to pass to the client"""

    model: str
    identifier: str = ''
    version: t.Optional[int] = None
    takes_context: bool = False
    encoder: t.Union[Encoder, str, None] = None
    client_kwargs: t.Dict[str, t.Any] = dc.field(default_factory=dict)

    type_id: t.ClassVar[str] = 'model'

    def __post_init__(self):
        self.identifier = self.identifier or self.model

    @property
    def child_components(self):
        if self.encoder is not None:
            return [('encoder', 'encoder')]
        return []


@dc.dataclass
class CohereEmbed(Cohere):
    """Cohere embedding predictor

    :param shape: The shape as ``tuple`` of the embedding.
    """

    shapes: t.ClassVar[t.Dict] = {'embed-english-v2.0': (4096,)}
    shape: t.Optional[t.Sequence[int]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.shape is None:
            self.shape = self.shapes[self.identifier]
        self.encoder = vector(self.shape)

    @retry
    def _predict_one(self, X: str, **kwargs):
        client = cohere.Client(get_key(KEY_NAME), **self.client_kwargs)
        e = client.embed(texts=[X], model=self.identifier, **kwargs)
        return e.embeddings[0]

    @retry
    async def _apredict_one(self, X: str, **kwargs):
        client = cohere.AsyncClient(get_key(KEY_NAME), **self.client_kwargs)
        e = await client.embed(texts=[X], model=self.identifier, **kwargs)
        await client.close()
        return e.embeddings[0]

    @retry
    def _predict_a_batch(self, texts: t.List[str], **kwargs):
        client = cohere.Client(get_key(KEY_NAME), **self.client_kwargs)
        out = client.embed(texts=texts, model=self.identifier, **kwargs)
        return [r for r in out.embeddings]

    @retry
    async def _apredict_a_batch(self, texts: t.List[str], **kwargs):
        client = cohere.AsyncClient(get_key(KEY_NAME), **self.client_kwargs)
        out = await client.embed(texts=texts, model=self.identifier, **kwargs)
        await client.close()
        return [r for r in out.embeddings]

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


@dc.dataclass
class CohereGenerate(Cohere):
    """Cohere realistic text generator (chat predictor)

    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to use to seed the response.
    """

    takes_context: bool = True
    prompt: str = ''

    @retry
    def _predict_one(self, X, context: t.Optional[t.List[str]] = None, **kwargs):
        if context is not None:
            X = format_prompt(X, self.prompt, context=context)
        client = cohere.Client(get_key(KEY_NAME), **self.client_kwargs)
        resp = client.generate(prompt=X, model=self.identifier, **kwargs)
        return resp.generations[0].text

    @retry
    async def _apredict_one(self, X, context: t.Optional[t.List[str]] = None, **kwargs):
        if context is not None:
            X = format_prompt(X, self.prompt, context=context)
        client = cohere.AsyncClient(get_key(KEY_NAME), **self.client_kwargs)
        resp = await client.generate(prompt=X, model=self.identifier, **kwargs)
        await client.close()
        return resp.generations[0].text

    def _predict(
        self, X, one: bool = True, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        if context:
            assert one, 'context only works with ``one=True``'
        if one:
            return self._predict_one(X, context=context, **kwargs)
        return [self._predict_one(msg) for msg in X]

    async def _apredict(
        self, X, one: bool = True, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        if context:
            assert one, 'context only works with ``one=True``'
        if one:
            return await self._apredict_one(X, context=context, **kwargs)
        return [await self._apredict_one(msg) for msg in X]
