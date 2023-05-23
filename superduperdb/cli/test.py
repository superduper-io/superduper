from . import command
from superduperdb.misc import run
from typer import Argument, Option
from typing import List
import sys

DIRS = ['superduperdb', 'tests']


@command(help='test')
def test(
    argv: List[str] = Argument(
        None,
        help='Additional arguments to pytest',
    ),
    check: bool = Option(
        False,
        '--check',
        '-c',
        help='If true, black and ruff fail without making changes',
    ),
    coverage: bool = Option(True, help='If True, run coverage on pytests'),
):
    black = ['black'] + check * ['--check'] + DIRS
    pytest = coverage * ['coverage', 'run', '-m'] + ['pytest'] + argv
    ruff = ['ruff'] + (not check) * ['--fix'] + DIRS

    try:
        run.run(black)
        run.run(ruff)
        run.run(pytest)
    except run.CalledProcessError:
        sys.exit('Tests failed')


if __name__ == '__main__':
    test(argv=[], check=True, coverage=True)
