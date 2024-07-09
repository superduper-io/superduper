from typer import Typer

__all__ = 'app', 'command'

CLI_NAME = 'superduper'

app = Typer(
    add_completion=False,
    context_settings={'help_option_names': ['--help', '-h']},
    help=f"""\

Usage: {CLI_NAME} [GLOBAL-FLAGS] [COMMAND] [COMMAND-FLAGS] [COMMAND-ARGS]

  Examples:

    $ {CLI_NAME} serve

    $ {CLI_NAME} local-cluster up

    $ {CLI_NAME} config
""",
)

command = app.command
