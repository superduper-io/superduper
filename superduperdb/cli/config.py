import json

from typer import Option

from superduperdb import CFG

from . import command


@command(help='Print all the SuperDuperDB configs as JSON')
def config(
    schema: bool = Option(
        False, '--schema', '-s', help='If set, print the JSON schema for the model'
    ),
):
    """Print all the SuperDuperDB configs as JSON.

    :param schema: If set, print the JSON schema for the model.
    """
    d = CFG.to_yaml() if schema else CFG.dict()
    if schema:
        print(CFG.to_yaml())
    else:
        print(json.dumps(d, indent=2))
