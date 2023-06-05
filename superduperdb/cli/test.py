from . import command
from superduperdb.misc import run
from typer import Argument, Option
from typing import List
import sys

DOCKER = 'docker compose -f tests/material/docker-compose.yml'.split()
DIRS = 'superduperdb tests'.split()

_HELP = """Run superduperdb tests, which are essentially
with extra flags
"""


@command(help=_HELP)
def test(
    argv: List[str] = Argument(
        None,
        help='Additional arguments to pytest',
    ),
    check: bool = Option(
        False,
        '--check',
        '-c',
        help='Only check files, do not change them',
    ),
    coverage: bool = Option(True, help='If True, run coverage on pytests'),
    down: bool = Option(False, help='If True, bring the docker down at the end'),
    dry_run: bool = Option(False, help='If True, print commands, do not execute them'),
):
    def run_all(cmd):
        try:
            for cmd in commands:
                if dry_run:
                    print('$', *cmd)
                else:
                    run.run(cmd)
        except run.CalledProcessError:
            sys.exit('Tests failed')

    run_all(
        DOCKER + ['up', 'mongodb', '-d'],
        ['black'] + check * ['--check'] + DIRS,
        ['ruff'] + (not check) * ['--fix'] + DIRS,
        ['mypy'],
        ['poetry', 'lock', '--no-update'] + check * ['--check'],
        coverage * ['coverage', 'run', '-m'] + ['pytest'] + argv,
    )
    if down:
        run_all(DOCKER + ['down'])


if __name__ == '__main__':
    test(argv=[], check=True, coverage=True)
