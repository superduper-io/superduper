import sys

import click

from superduper.cli import app, main

__all__ = (
    'app',
    'main',
)


def run():
    """Entrypoint for the CLI.

    This is the function that is called when the user runs `python -m superduper`.
    """
    try:
        app(standalone_mode=False)
    except click.ClickException as e:
        return f'{e.__class__.__name__}: {e.message}'
    except click.Abort:
        return 'Aborted'


if __name__ == '__main__':
    sys.exit(run())
