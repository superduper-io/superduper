import dataclasses as dc
import typing as t
from functools import wraps

from superduperdb.components.component import Component
from superduperdb.components.model import _Predictor
from superduperdb.ext.utils import format_prompt


@dc.dataclass
class LLMBase(Component, _Predictor):
    max_tokens: int = 512
    temperature: float = 0.0
    prompt_template: str = "{input}"
    prompt_func: t.Optional[t.Callable] = dc.field(default=None)

    def __post_init__(self):
        super().__post_init__()
        self.takes_context = True
        assert "{input}" in self.prompt_template, "Template must contain {input}"

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
class LLMAPI(LLMBase):
    api_url: str = dc.field(default="")

    def __post_init__(self):
        super().__post_init__()
        assert self.api_url, "api_url can not be empty"


def ensure_initialized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_is_initialized') or not self._is_initialized:
            print("Start loading model")
            self.init()
            self._is_initialized = True
        return func(self, *args, **kwargs)

    return wrapper


@dc.dataclass
class LLMModel(LLMBase):
    model_name: str = dc.field(default="")
    on_ray: bool = False
    ray_config: dict = dc.field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        assert self.model_name, "model_name can not be empty"
        self._is_initialized = False

    @ensure_initialized
    def _predict(self, X, one: bool = False, **kwargs):
        return super()._predict(X, one, **kwargs)
