import dataclasses as dc
import typing as t

import anthropic
from anthropic import APIConnectionError, APIError, APIStatusError, APITimeoutError

from superduperdb.container.component import Component
from superduperdb.container.encoder import Encoder
from superduperdb.container.model import PredictMixin
from superduperdb.ext.utils import format_prompt, get_key
from superduperdb.misc.retry import Retry

retry = Retry(
    exception_types=(APIConnectionError, APIError, APIStatusError, APITimeoutError)
)

KEY_NAME = 'ANTHROPIC_API_KEY'


@dc.dataclass
class Anthropic(Component, PredictMixin):
    """Anthropic predictor.

    :param model: The model to use, e.g. ``'claude-2'``.
    :param identifier: The identifier to use, e.g. ``'my-model'``.
    :param version: The version to use, e.g. ``0`` (leave empty)
    :param takes_context: Whether the model takes context into account.
    :param encoder: The encoder identifier.
    :param type_id: A unique name for the class
    :param client_kwargs: Keyword arguments to pass to the client
    """

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
    def child_components(self) -> t.Sequence[t.Tuple[str, str]]:
        if self.encoder is not None:
            return [('encoder', 'encoder')]
        return []


@dc.dataclass
class AnthropicCompletions(Anthropic):
    """Cohere completions (chat) predictor.

    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to use to seed the response.
    """

    takes_context: bool = True
    prompt: str = ''

    @retry
    def _predict_one(self, X, context: t.Optional[t.List[str]] = None, **kwargs):
        if context is not None:
            X = format_prompt(X, self.prompt, context=context)
        client = anthropic.Anthropic(api_key=get_key(KEY_NAME), **self.client_kwargs)
        resp = client.completions.create(prompt=X, model=self.identifier, **kwargs)
        return resp.completion

    @retry
    async def _apredict_one(self, X, context: t.Optional[t.List[str]] = None, **kwargs):
        if context is not None:
            X = format_prompt(X, self.prompt, context=context)
        client = anthropic.AsyncAnthropic(
            api_key=get_key(KEY_NAME), **self.client_kwargs
        )
        resp = await client.completions.create(
            prompt=X, model=self.identifier, **kwargs
        )
        return resp.completion

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
