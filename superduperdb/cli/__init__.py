from typer import Typer

from .. import ICON

__all__ = 'app', 'command'

CLI_NAME = 'superduperdb'

app = Typer(
    add_completion=False,
    context_settings={'help_option_names': ['--help', '-h']},
    help=f"""\
{ICON} {CLI_NAME} {ICON}

Usage: {CLI_NAME} [GLOBAL-FLAGS] [COMMAND] [COMMAND-FLAGS] [COMMAND-ARGS]

  Examples:

    $ {CLI_NAME} serve

    $ {CLI_NAME} local-cluster users documents products

    $ {CLI_NAME} config
""",
)

command = app.command
