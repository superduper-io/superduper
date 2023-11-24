import sys

import click

from superduperdb.cli import app, config, info
from superduperdb.cli.serve import cdc, local_cluster, vector_search

__all__ = 'config', 'info', 'local_cluster', 'vector_search', 'cdc'


def run():
    """
    Entrypoint for the CLI. This is the function that is called when the
    user runs `python -m superduperdb`.
    """
    try:
        app(standalone_mode=False)
    except click.ClickException as e:
        return f'{e.__class__.__name__}: {e.message}'
    except click.Abort:
        return 'Aborted'


if __name__ == '__main__':
    sys.exit(run())
