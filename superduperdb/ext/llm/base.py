import abc
import dataclasses as dc
import typing as t

from superduperdb.components.component import Component
from superduperdb.components.model import _Predictor
from superduperdb.ext.utils import format_prompt


@dc.dataclass
class LLMModel(Component, _Predictor, metaclass=abc.ABCMeta):
    model_name: t.Optional[str] = dc.field(default="", repr=False)
    max_tokens: int = 512
    temperature: float = 0.0
    prompt_template: str = "{input}"
    prompt_func: t.Optional[t.Callable] = dc.field(default=None, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self.takes_context = True
        assert "{input}" in self.prompt_template, "Template must contain {input}"

    @abc.abstractmethod
    def _predict_one(self, X, **kwargs):
        pass

    def _predict(self, X, one: bool = False, **kwargs):
        if one:
            x = self.format_prompt(X, **kwargs)
            return self._predict_one(x, **kwargs)
        else:
            xs = [self.format_prompt(x, **kwargs) for x in X]
            return [self._predict_one(x, **kwargs) for x in xs]

    def format_prompt(self, x, **kwargs):
        context = kwargs.get("context")
        if self.prompt_func is not None:
            return self.prompt_func(x, context=context)

        return format_prompt(x, self.prompt_template, context)


@dc.dataclass
class LLMAPI(LLMModel):
    api_url: str = dc.field(default="", repr=False)
