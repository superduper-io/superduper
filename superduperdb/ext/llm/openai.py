import dataclasses as dc

from superduperdb.misc.annotations import requires_packages

requires_packages(['openai', '1.1.2', None], ['httpx'])
from httpx import ResponseNotRead
from openai import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from superduperdb.ext.llm.base import BaseOpenAI
from superduperdb.misc.retry import Retry

retry = Retry(
    exception_types=(
        APITimeoutError,
        RateLimitError,
        InternalServerError,
        ResponseNotRead,
    )
)


@dc.dataclass
class OpenAI(BaseOpenAI):
    """
    OpenAI chat completion predictor.
    {parent_doc}
    """

    __doc__ = __doc__.format(parent_doc=BaseOpenAI.__doc__)

    def __post_init__(self, artifacts):
        """Set model name."""
        # only support chat mode
        self.chat = True
        super().__post_init__(artifacts)

    @retry
    def get_model_set(self):
        return super().get_model_set()

    @retry
    def _generate(self, *args, **kwargs) -> str:
        return super()._generate(*args, **kwargs)
