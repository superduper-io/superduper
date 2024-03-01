import dataclasses as dc
import inspect
import typing as t

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
            return format_prompt(x, self.prompt_template, kwargs.pop("context", None))
        else:
            return x
