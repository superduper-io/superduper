from . import command
from typer import Argument, Option
from typing import List


@command(help='serve')
def serve(
    arg: str = Argument(
        'arg',
        help='First argument',
    ),
    argv: List[str] = Argument(
        None,
        help='Additional arguments',
    ),
    opt: bool = Option(False, '--opt', '-o', help='An option'),
):
    pass
