import sys

import click

from superduperdb.cli import app, config, info, vector_search
from superduperdb.cli.serve import local_cluster, serve

__all__ = 'config', 'info', 'local_cluster', 'serve', 'vector_search'


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
