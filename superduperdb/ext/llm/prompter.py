import dataclasses as dc
import inspect
import typing as t

from superduperdb.components.model import QueryModel
from superduperdb.ext.utils import format_prompt


@dc.dataclass
class Prompter:
    prompt_template: str = "{input}"
    prompt_func: t.Optional[t.Callable] = dc.field(default=None)

    def __call__(self, x: t.Any, **kwargs):
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


@dc.dataclass(kw_only=True)
class RetrievalPrompt(QueryModel):
    """
    This function creates a prompt based on data
    recalled from the database and a pre-specified
    question:
    """

    prompt_explanation: str = PROMPT_EXPLANATION
    prompt_introduction: str = PROMPT_INTRODUCTION
    join: str = "\n---\n"
    ui_schema: t.ClassVar[t.List[t.Dict]] = [
        {'name': 'prompt_explanation', 'type': 'str', 'default': PROMPT_EXPLANATION},
        {'name': 'prompt_introduction', 'type': 'str', 'default': PROMPT_INTRODUCTION},
        {'name': 'join', 'type': 'str', 'default': "\n---\n"},
    ]

    def __post_init__(self, artifacts):
        assert len(self.select.variables) == 1
        assert next(iter(self.select.variables)).value == 'prompt'
        return super().__post_init__(artifacts)

    @property
    def inputs(self):
        return super().inputs

    def predict_one(self, prompt):
        out = super().predict_one(prompt=prompt)
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
