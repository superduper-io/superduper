import dataclasses as dc
import inspect
import typing as t

from superduper.components.model import QueryModel
from superduper.misc.utils import format_prompt


@dc.dataclass
class Prompter:
    """Prompt the user for input.

    This function prompts the user for input based on a
    template string and a function which formats the
    prompt.

    :param prompt_template: The template string for the prompt.
    :param prompt_func: The function which formats the prompt.
    """

    prompt_template: str = "{input}"
    prompt_func: t.Optional[t.Callable] = dc.field(default=None)

    def __call__(self, x: t.Any, **kwargs):
        """Format the prompt.

        :param x: The input to format the prompt.
        :param kwargs: The keyword arguments to pass to the prompt function.
        """
        if self.prompt_func is not None:
            sig = inspect.signature(self.prompt_func)
            new_kwargs = {}
            for k, v in kwargs.items():
                if k in sig.parameters:
                    new_kwargs[k] = v
            return self.prompt_func(x, **new_kwargs)

        if isinstance(x, str):
            return format_prompt(x, self.prompt_template, kwargs.get("context", None))
        else:
            return x


PROMPT_EXPLANATION = (
    "HERE ARE SOME FACTS SEPARATED BY '---' IN OUR DATA "
    "REPOSITORY WHICH WILL HELP YOU ANSWER THE QUESTION."
)

PROMPT_INTRODUCTION = (
    "HERE IS THE QUESTION WHICH YOU SHOULD ANSWER BASED ONLY ON THE PREVIOUS FACTS:"
)


class RetrievalPrompt(QueryModel):
    """Retrieve a prompt based on data recalled from the database.

    This function creates a prompt based on data
    recalled from the database and a pre-specified
    question:

    :param prompt_explanation: The explanation of the prompt.
    :param prompt_introduction: The introduction of the prompt.
    :param join: The string to join the facts.
    """

    prompt_explanation: str = PROMPT_EXPLANATION
    prompt_introduction: str = PROMPT_INTRODUCTION
    join: str = "\n---\n"

    def postinit(self):
        super().postinit()
        assert 'prompt' in self.select.variables

    @property
    def inputs(self):
        """The inputs of the model."""
        return super().inputs

    def predict(self, prompt):
        """Create a prompt based on the facts and the question.

        :param prompt: The prompt to answer the question.
        """
        out = super().predict(prompt=prompt)
        prompt = (
            self.prompt_explanation
            + self.join
            + self.join.join(out)
            + self.join
            + self.prompt_introduction
            + self.join
            + prompt
        )
        return prompt
