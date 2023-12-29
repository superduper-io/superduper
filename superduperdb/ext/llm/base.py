import abc
import asyncio
import dataclasses as dc
import inspect
import typing
from functools import wraps
from logging import WARNING, getLogger
from typing import Any, Callable, List, Optional, Union

from superduperdb import logging
from superduperdb.components.component import Component
from superduperdb.components.model import _Predictor
from superduperdb.ext.utils import format_prompt

if typing.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer

# Disable httpx info level logging
getLogger("httpx").setLevel(WARNING)


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
    """
    :param identifier: The identifier for the model.
    :param max_tokens: The maximum number of tokens to generate.
    :param temperature: The temperature to use for generation.
    :param prompt_template: The template to use for the prompt.
    :param prompt_func: The function to use for the prompt.
    :param max_batch_size: The maximum batch size to use for batch generation.
    """

    max_tokens: int = 64
    temperature: float = 0.0
    prompt_template: str = "{input}"
    prompt_func: Optional[Callable] = dc.field(default=None)
    max_batch_size: Optional[int] = 64

    def __post_init__(self):
        super().__post_init__()
        self.takes_context = True
        self.identifier = self.identifier.replace("/", "-")
        assert "{input}" in self.prompt_template, "Template must contain {input}"

    def to_call(self, X, *args, **kwargs):
        raise NotImplementedError

    def post_create(self, db: "Datalayer") -> None:
        # TODO: Do not make sense to add this logic here,
        # Need a auto DataType to handle this
        from superduperdb.backends.ibis.data_backend import IbisDataBackend
        from superduperdb.backends.ibis.field_types import dtype

        if isinstance(db.databackend, IbisDataBackend) and self.encoder is None:
            self.encoder = dtype('str')

        # since then the `.add` clause is not necessary
        output_component = db.databackend.create_model_table_or_collection(self)  # type: ignore[arg-type]
        if output_component is not None:
            db.add(output_component)

    @abc.abstractmethod
    def init(self):
        ...

    @abc.abstractmethod
    def _generate(self, prompt: str, **kwargs: Any) -> str:
        ...

    def _batch_generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """
        Base method to batch generate text from a list of prompts.
        If the model can run batch generation efficiently, pls override this method.
        """
        return [self._generate(prompt, **kwargs) for prompt in prompts]

    @ensure_initialized
    def _predict(self, X: Union[str, List[str]], one: bool = False, **kwargs: Any):
        if isinstance(X, str):
            x = self.format_prompt(X, **kwargs)
            return self._generate(x, **kwargs)
        else:
            xs = [self.format_prompt(x, **kwargs) for x in X]
            return self._batch_generate(xs, **kwargs)

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
    """
    :param api_url: The URL for the API.
    {parent_doc}
    """

    __doc__ = __doc__.format(parent_doc=_BaseLLM.__doc__)

    api_url: str = dc.field(default="")

    def __post_init__(self):
        super().__post_init__()
        assert self.api_url, "api_url can not be empty"

    def init(self):
        pass


@dc.dataclass
class BaseOpenAI(_BaseLLM):
    """
    :param openai_api_base: The base URL for the OpenAI API.
    :param openai_api_key: The API key to use for the OpenAI API.
    :param model_name: The name of the model to use.
    :param chat: Whether to use the chat API or the completion API. Defaults to False.
    :param system_prompt: The prompt to use for the system.
    :param user_role: The role to use for the user.
    :param system_role: The role to use for the system.
    {parent_doc}
    """

    __doc__ = __doc__.format(parent_doc=_BaseLLM.__doc__)

    identifier: str = dc.field(default="")
    openai_api_base: Optional[str] = None
    openai_api_key: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"
    chat: bool = True
    system_prompt: Optional[str] = None
    user_role: str = "user"
    system_role: str = "system"

    def __post_init__(self):
        self.identifier = self.identifier or self.model_name
        super().__post_init__()

    def init(self):
        try:
            from openai import AsyncClient, OpenAI
        except ImportError:
            raise Exception("You must install openai with command 'pip install openai'")

        params = {
            "api_key": self.openai_api_key,
            "base_url": self.openai_api_base,
        }

        self.client = OpenAI(**params)
        self.aclient = AsyncClient(**params)
        model_list = self.client.models.list()
        model_set = sorted({model.id for model in model_list.data})
        assert (
            self.model_name in model_set
        ), f"model_name {self.model_name} is not in model_set {model_set}"

    def _generate(self, prompt: str, **kwargs: Any) -> str:
        if self.chat:
            return self._chat_generate(prompt, **kwargs)
        else:
            return self._prompt_generate(prompt, **kwargs)

    def _batch_generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """
        Use asyncio to batch generate text from a list of prompts.
        """
        return asyncio.run(self._async_batch_generate(prompts, **kwargs))

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

    async def _async_generate(self, semaphore, prompt: str, **kwargs) -> str:
        async with semaphore:
            if self.chat:
                return await self._async_chat_generate(prompt, **kwargs)
            else:
                return await self._async_prompt_generate(prompt, **kwargs)

    async def _async_prompt_generate(self, prompt: str, **kwargs) -> str:
        """
        Async method to generate a completion for a given prompt with prompt format.
        """
        completion = await self.aclient.completions.create(
            model=self.model_name,
            prompt=prompt,
            **self.get_kwargs(self.aclient.completions.create, **kwargs),
        )
        return completion.choices[0].text

    async def _async_chat_generate(self, content: str, **kwargs) -> str:
        """
        Async method to generate a completion for a given prompt with chat format.
        """
        messages = kwargs.get("messages", [])

        if self.system_prompt:
            messages = [
                {"role": self.system_role, "content": self.system_prompt}
            ] + messages

        messages.append({"role": self.user_role, "content": content})
        completion = await self.aclient.chat.completions.create(
            messages=messages,
            model=self.model_name,
            **self.get_kwargs(self.aclient.chat.completions.create, **kwargs),
        )
        return completion.choices[0].message.content

    async def _async_batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Async method to concurrently process multiple prompts.
        """
        semaphore = asyncio.Semaphore(self.max_batch_size or len(prompts))
        tasks = [
            self._async_generate(semaphore, prompt, **kwargs) for prompt in prompts
        ]
        return await asyncio.gather(*tasks)


@dc.dataclass
class BaseLLMModel(_BaseLLM):
    """
    :param model_name: The name of the model to use.
    :param on_ray: Whether to run the model on Ray.
    :param ray_config: The Ray config to use.
    {parent_doc}
    """

    __doc__ = __doc__.format(parent_doc=_BaseLLM.__doc__)

    model_name: str = dc.field(default="")
    on_ray: bool = False
    ray_config: dict = dc.field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        assert self.model_name, "model_name can not be empty"
        self._is_initialized = False
