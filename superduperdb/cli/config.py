from . import command
from superduperdb import CFG
from typer import Option


@command(help='Print all the SuperDuperDB configs as JSON')
def config(
    schema: bool = Option(
        False, '--schema', '-s', help='If set, print the JSON schema for the model'
    ),
):
    json = CFG.schema_json if schema else CFG.json
    print(json(indent=2))
