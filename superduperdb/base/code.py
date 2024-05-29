import dataclasses as dc
import inspect

from superduperdb.base.leaf import Leaf
from superduperdb.misc.annotations import merge_docstrings

template = """from superduperdb import code

@code
{definition}"""


@merge_docstrings
@dc.dataclass(kw_only=True)
class Code(Leaf):
    """A class to store remote code.

    This class stores remote code that can be executed on a remote server.

    :param code: The code to store.
    """

    code: str
    identifier: str = ''

    @staticmethod
    def from_object(obj):
        """Create a Code object from a callable object.

        :param obj: The object to create the Code object from.
        """
        code = inspect.getsource(obj)
        mini_module = template.format(
            definition=code,
        )
        print(mini_module)
        return Code(code=mini_module)

    def __post_init__(self, db):
        super().__post_init__(db)
        namespace = {}
        exec(self.code, namespace)
        remote_code = next(
            (obj for obj in namespace.values() if hasattr(obj, 'is_remote_code')),
            None,
        )
        if remote_code is None:
            raise ValueError('No remote code found in the provided code')
        self.object = remote_code

    def __call__(self, *args, **kwargs):
        """
        Call the code-object on parameters.

        :param args: The positional arguments to pass to the code
        :param kwargs: The keyword arguments to pass to the code
        """
        return self.object(*args, **kwargs)

    def unpack(self):
        """Unpack the code object. Does nothing."""
        return self
