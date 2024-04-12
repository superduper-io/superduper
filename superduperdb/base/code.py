import dataclasses as dc
import inspect
import typing as t

from superduperdb.base.serializable import Serializable

template = \
"""from superduperdb import code
@code
{definition}"""

default = template.format(definition='def my_code(*args, **kwargs):\n    ...\n    return\n')


@dc.dataclass
class Code(Serializable):
    code: str
    default: t.ClassVar[str] = default

    @staticmethod
    def from_object(obj):
        code = inspect.getsource(obj)

        mini_module = template.format(
            definition=code,
        )
        print(mini_module)
        return Code(mini_module)

    def __post_init__(self):
        namespace = {}
        exec(self.code, namespace)
        remote_code = next(
            (obj for obj in namespace.values() if hasattr(obj, 'is_remote_code')),
            None,
        )
        if remote_code is None:
            raise ValueError('No remote code found in the provided code')
        self.object = remote_code

    def unpack(self, db=None):
        return self.object