import dataclasses as dc
import typing as t

import anthropic
from anthropic import APIConnectionError, APIError, APIStatusError, APITimeoutError

from superduperdb.backends.ibis.data_backend import IbisDataBackend
from superduperdb.backends.ibis.field_types import dtype
from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.base.datalayer import Datalayer
from superduperdb.components.model import APIModel
from superduperdb.ext.utils import format_prompt, get_key
from superduperdb.misc.retry import Retry

retry = Retry(
    exception_types=(APIConnectionError, APIError, APIStatusError, APITimeoutError)
)

KEY_NAME = 'ANTHROPIC_API_KEY'


@dc.dataclass(kw_only=True)
class Anthropic(APIModel):
    """Anthropic predictor."""

    client_kwargs: t.Dict[str, t.Any] = dc.field(default_factory=dict)

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)
        self.identifier = self.identifier or self.model


@dc.dataclass(kw_only=True)
class AnthropicCompletions(Anthropic):
    """Cohere completions (chat) predictor.

    :param takes_context: Whether the model takes context into account.
    :param prompt: The prompt to use to seed the response.
    """

    prompt: str = ''

    def pre_create(self, db: Datalayer) -> None:
        super().pre_create(db)
        if isinstance(db.databackend, IbisDataBackend) and self.datatype is None:
            self.datatype = dtype('str')

    @retry
    def predict_one(
        self, prompt: str, context: t.Optional[t.List[str]] = None, **kwargs
    ):
        if context is not None:
            prompt = format_prompt(prompt, self.prompt, context=context)
        client = anthropic.Anthropic(api_key=get_key(KEY_NAME), **self.client_kwargs)
        resp = client.completions.create(prompt=prompt, model=self.identifier, **kwargs)
        return resp.completion

    def predict(self, dataset: t.Union[t.List, QueryDataset]) -> t.List:
        return [self.predict_one(dataset[i]) for i in range(len(dataset))]
