import abc
import dataclasses as dc
import inspect
from functools import wraps
from typing import Any, Callable, List, Optional, Union

from superduperdb import logging
from superduperdb.components.component import Component
from superduperdb.components.model import _Predictor
from superduperdb.ext.utils import format_prompt


def ensure_initialized(func):
    """Decorator to ensure that the model is initialized before calling the function"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_is_initialized") or not self._is_initialized:
            model_message = f"{self.__class__.__name__} : {self.identifier}"
            logging.info(f"Initializing {model_message}")
            self.init()
            self._is_initialized = True
            logging.info(f"Initialized  {model_message} successfully")
        return func(self, *args, **kwargs)

    return wrapper


@dc.dataclass
class _BaseLLM(Component, _Predictor, metaclass=abc.ABCMeta):
    max_tokens: int = 64
    temperature: float = 0.0
    prompt_template: str = "{input}"
    prompt_func: Optional[Callable] = dc.field(default=None)

    def __post_init__(self):
        super().__post_init__()
        self.takes_context = True
        assert "{input}" in self.prompt_template, "Template must contain {input}"

    @abc.abstractmethod
    def init(self):
        ...

    @abc.abstractmethod
    def _generate(self, prompt: str, **kwargs: Any) -> str:
        ...

    @ensure_initialized
    def _predict(self, X: Union[str, List[str]], one: bool = False, **kwargs: Any):
        one = isinstance(X, str) or one
        if one:
            x = self.format_prompt(X, **kwargs)
            return self._generate(x, **kwargs)
        else:
            xs = [self.format_prompt(x, **kwargs) for x in X]
            return [self._generate(x, **kwargs) for x in xs]

    def format_prompt(self, x, **kwargs):
        """
        Format a prompt with the given prompt function or template.
        :param x: The input to format.
        :param kwargs: Any additional arguments to pass to the prompt function.
        """
        if self.prompt_func is not None:
            return self.prompt_func(x, **self.get_kwargs(self.prompt_func, **kwargs))

        return format_prompt(x, self.prompt_template, kwargs.get("context", None))

    def get_kwargs(self, func, **kwargs):
        """
        Get kwargs and object attributes that are in the function signature
        :param func (Callable): function to get kwargs for
        :param kwargs (dict): kwargs to filter
        """
        sig = inspect.signature(func)
        new_kwargs = {}
        for k, v in {**self.dict(), **kwargs}.items():
            if k in sig.parameters:
                new_kwargs[k] = v
        return new_kwargs


@dc.dataclass
class BaseLLMAPI(_BaseLLM):
    """Base class for LLMs that use an API"""

    api_url: str = dc.field(default="")

    def __post_init__(self):
        super().__post_init__()
        assert self.api_url, "api_url can not be empty"

    def init(self):
        pass


@dc.dataclass
class BaseOpenAI(_BaseLLM):
    """
    Base class for LLMs that use OpenAI API
    Use openai-python package to interact with OpenAI format API
    """

    identifier: str = dc.field(default="")
    openai_api_base: Optional[str] = None
    openai_api_key: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"
    chat: bool = False
    system_prompt: Optional[str] = None
    user_role: str = "user"
    system_role: str = "system"

    def __post_init__(self):
        self.identifier = self.identifier or self.model_name
        super().__post_init__()

    def init(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise Exception("You must install openai with command 'pip install openai'")

        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.openai_api_base)
        model_list = self.client.models.list()
        model_set = {model.id for model in model_list.data}
        assert (
            self.model_name in model_set
        ), f"model_name {self.model_name} is not in model_set {model_set}"

    def _generate(self, prompt: str, **kwargs: Any) -> str:
        if self.chat:
            return self._chat_generate(prompt, **kwargs)
        else:
            return self._prompt_generate(prompt, **kwargs)

    def _prompt_generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a completion for a given prompt with prompt format.
        """
        completion = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            **self.get_kwargs(self.client.completions.create, **kwargs),
        )
        return completion.choices[0].text

    def _chat_generate(self, content: str, **kwargs: Any) -> str:
        """
        Generate a completion for a given prompt with chat format.
        :param prompt: The prompt to generate a completion for.
        :param kwargs: Any additional arguments to pass to the API.
        """
        messages = kwargs.get("messages", [])

        if self.system_prompt:
            messages = [
                {"role": self.system_role, "content": self.system_prompt}
            ] + messages

        messages.append({"role": self.user_role, "content": content})
        completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            **self.get_kwargs(self.client.chat.completions.create, **kwargs),
        )
        return completion.choices[0].message.content


@dc.dataclass
class BaseLLMModel(_BaseLLM):
    model_name: str = dc.field(default="")
    on_ray: bool = False
    ray_config: dict = dc.field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        assert self.model_name, "model_name can not be empty"
        self._is_initialized = False
