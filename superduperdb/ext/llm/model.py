import concurrent.futures
import dataclasses as dc
import inspect
import typing as t
from functools import reduce
from logging import WARNING, getLogger

from superduperdb import logging
from superduperdb.backends.query_dataset import QueryDataset
from superduperdb.components.component import ensure_initialized
from superduperdb.components.model import Model
from superduperdb.ext.llm.prompter import Prompter

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer

# Disable httpx info level logging
getLogger("httpx").setLevel(WARNING)


class BaseLLM(Model):
    """Base class for LLM models.

    :param prompt: The template to use for the prompt.
    :param prompt_func: The function to use for the prompt.
    :param max_batch_size: The maximum batch size to use for batch generation.
    """

    prompt: str = "{input}"
    prompt_func: t.Optional[t.Callable] = dc.field(default=None)
    max_batch_size: t.Optional[int] = 4
    signature: str = 'singleton'

    def __post_init__(self, db, artifacts):
        super().__post_init__(db, artifacts)
        self.takes_context = True
        self.identifier = self.identifier.replace("/", "-")

    def post_create(self, db: "Datalayer") -> None:
        """Post create method for the model.

        :param db: The datalayer to use for the model.
        """
        # TODO: Do not make sense to add this logic here,
        # Need a auto DataType to handle this
        from superduperdb.backends.ibis.data_backend import IbisDataBackend
        from superduperdb.backends.ibis.field_types import dtype

        if isinstance(db.databackend, IbisDataBackend) and self.datatype is None:
            self.datatype = dtype("str")

        super().post_create(db)

    def _generate(self, prompt: str, **kwargs: t.Any):
        raise NotImplementedError

    def _batch_generate(self, prompts: t.List[str], **kwargs) -> t.List[str]:
        """Base method to batch generate text from a list of prompts.

        If the model can run batch generation efficiently, pls override this method.

        :param prompts: The list of prompts to generate text from.
        :param kwargs: The keyword arguments to pass to the prompt function and
                        the llm model.
        """
        return [self._generate(prompt, **self.predict_kwargs) for prompt in prompts]

    @ensure_initialized
    def predict(self, X: t.Union[str, dict[str, str]], context=None, **kwargs):
        """Generate text from a single input.

        :param X: The input to generate text from.
        :param context: The context to use for the prompt.
        :param kwargs: The keyword arguments to pass to the prompt function and
                        the llm model.
        """
        x = self.prompter(X, context=context, **kwargs)
        return self._generate(x, **kwargs)

    @ensure_initialized
    def predict_batches(
        self, dataset: t.Union[t.List, QueryDataset], **kwargs
    ) -> t.Sequence:
        """Generate text from a dataset.

        :param dataset: The dataset to generate text from.
        :param kwargs: The keyword arguments to pass to the prompt function and
                        the llm model.

        """
        xs = [self.prompter(dataset[i], **kwargs) for i in range(len(dataset))]
        kwargs.pop("context", None)
        return self._batch_generate(xs, **kwargs)

    def get_kwargs(self, func: t.Callable, *kwargs_list):
        """Get kwargs and object attributes that are in the function signature.

        :param func: function to get kwargs for
        :param *kwargs_list: kwargs to filter
        """
        total_kwargs = reduce(lambda x, y: {**y, **x}, [self.dict(), *kwargs_list])
        sig = inspect.signature(func)
        new_kwargs = {}
        for k, v in total_kwargs.items():
            if k in sig.parameters:
                new_kwargs[k] = v
        return new_kwargs

    @property
    def prompter(self):
        """Return a prompter for the model."""
        return Prompter(self.prompt, self.prompt_func)


class BaseLLMAPI(BaseLLM):
    """Base class for LLM models with an API.

    :param api_url: The URL for the API.
    """

    api_url: str = dc.field(default="")

    def init(self):
        """Initialize the model."""
        pass

    def _generate_wrapper(self, prompt: str, **kwargs: t.Any) -> str:
        """Wrapper for the _generate method to handle exceptions."""
        try:
            return self._generate(prompt, **kwargs)
        except Exception as e:
            logging.error(f"Error generating response for prompt '{prompt}': {e}")
            return ""

    def _batch_generate(self, prompts: t.List[str], **kwargs: t.Any) -> t.List[str]:
        """
        Base method to batch generate text from a list of prompts using multi-threading.

        Handles exceptions in _generate method.

        :param prompts: The list of prompts to generate text from.
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
