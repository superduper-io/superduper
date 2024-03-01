import abc
import concurrent.futures
import dataclasses as dc
import inspect
import typing
from functools import reduce
from logging import WARNING, getLogger
from typing import Any, Callable, List, Optional, Sequence, Union

from superduperdb import logging
from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.components.component import ensure_initialized
from superduperdb.components.model import _Predictor
from superduperdb.ext.llm.utils import Prompter

if typing.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer

# Disable httpx info level logging
getLogger("httpx").setLevel(WARNING)


@dc.dataclass
class _BaseLLM(_Predictor, metaclass=abc.ABCMeta):
    """
    :param prompt_template: The template to use for the prompt.
    :param prompt_func: The function to use for the prompt.
    :param max_batch_size: The maximum batch size to use for batch generation.
    :param predict_kwargs: Parameters used during inference.
    """

    prompt_template: str = "{input}"
    prompt_func: Optional[Callable] = dc.field(default=None)
    max_batch_size: Optional[int] = 4

    def __post_init__(self, artifacts):
        super().__post_init__(artifacts)
        self.takes_context = True
        self.identifier = self.identifier.replace("/", "-")
        self.prompter = Prompter(self.prompt_template, self.prompt_func)
        assert "{input}" in self.prompt_template, "Template must contain {input}"

    def to_call(self, X, *args, **kwargs):
        raise NotImplementedError

    def post_create(self, db: "Datalayer") -> None:
        # TODO: Do not make sense to add this logic here,
        # Need a auto DataType to handle this
        from superduperdb.backends.ibis.data_backend import IbisDataBackend
        from superduperdb.backends.ibis.field_types import dtype

        if isinstance(db.databackend, IbisDataBackend) and self.datatype is None:
            self.datatype = dtype("str")

        super().post_create(db)

    @abc.abstractmethod
    def init(self):
        ...

    @abc.abstractmethod
    def _generate(self, prompt: str, **kwargs: Any) -> str:
        ...

    def _batch_generate(self, prompts: List[str]) -> List[str]:
        """
        Base method to batch generate text from a list of prompts.
        If the model can run batch generation efficiently, pls override this method.
        """
        return [self._generate(prompt, **self.predict_kwargs) for prompt in prompts]

    @ensure_initialized
    def predict_one(self, X: Union[str, dict[str, str]], **kwargs):
        x = self.prompter(X)
        return self._generate(x, **kwargs)

    @ensure_initialized
    def predict(self, dataset: Union[List, QueryDataset]) -> Sequence:
        xs = [self.prompter(dataset[i]) for i in range(len(dataset))]
        return self._batch_generate(xs)

    def get_kwargs(self, func, *kwargs_list):
        """
        Get kwargs and object attributes that are in the function signature
        :param func (Callable): function to get kwargs for
        :param kwargs (list of dict): kwargs to filter
        """

        total_kwargs = reduce(lambda x, y: {**y, **x}, [self.dict(), *kwargs_list])
        sig = inspect.signature(func)
        new_kwargs = {}
        for k, v in total_kwargs.items():
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

    def init(self):
        pass

    def _generate_wrapper(self, prompt: str, **kwargs: Any) -> str:
        """
        Wrapper for the _generate method to handle exceptions.
        """
        try:
            return self._generate(prompt, **kwargs)
        except Exception as e:
            logging.error(f"Error generating response for prompt '{prompt}': {e}")
            return ""

    def _batch_generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """
        Base method to batch generate text from a list of prompts using multi-threading.
        Handles exceptions in _generate method.
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_batch_size
        ) as executor:
            results = list(
                executor.map(
                    lambda prompt: self._generate_wrapper(prompt, **kwargs), prompts
                )
            )

        return results


@dc.dataclass
class BaseOpenAI(BaseLLMAPI):
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
    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_key: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"
    chat: bool = True
    system_prompt: Optional[str] = None
    user_role: str = "user"
    system_role: str = "system"

    def __post_init__(self, artifacts):
        self.api_url = self.openai_api_base
        self.identifier = self.identifier or self.model_name
        super().__post_init__(artifacts)

    def init(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise Exception("You must install openai with command 'pip install openai'")

        params = {
            "api_key": self.openai_api_key,
            "base_url": self.openai_api_base,
        }

        self.client = OpenAI(**params)
        model_set = self.get_model_set()
        assert (
            self.model_name in model_set
        ), f"model_name {self.model_name} is not in model_set {model_set}"

    def get_model_set(self):
        model_list = self.client.models.list()
        return sorted({model.id for model in model_list.data})

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
            **self.get_kwargs(
                self.client.completions.create, kwargs, self.predict_kwargs
            ),
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
            **self.get_kwargs(
                self.client.chat.completions.create, kwargs, self.predict_kwargs
            ),
        )
        return completion.choices[0].message.content


@dc.dataclass
class BaseLLMModel(_BaseLLM):
    """
    :param model_name: The name of the model to use.
    :param on_ray: Whether to run the model on Ray.
    :param ray_config: The Ray config to use.
    :param ray_addredd: The address of the ray cluster.
    {parent_doc}
    """

    __doc__ = __doc__.format(parent_doc=_BaseLLM.__doc__)

    identifier: str = dc.field(default="")
    model_name: str = dc.field(default="")
    on_ray: bool = False
    ray_address: Optional[str] = None
    ray_config: dict = dc.field(default_factory=dict)

    def __post_init__(self):
        self.identifier = self.identifier or self.model_name
        assert self.model_name, "model_name can not be empty"
        self._is_initialized = False
        super().__post_init__()
