import dataclasses as dc
import inspect
import typing as t

from superduperdb.base.leaf import Leaf

template = """from superduperdb import code

@code
{definition}"""

default = template.format(definition='def my_code(x):\n    return x\n')


@dc.dataclass(kw_only=True)
class Code(Leaf):
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
