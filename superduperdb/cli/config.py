from . import command
from superduperdb import CFG
import json


@command(help='Print all the SuperDuperDB configs as JSON')
def config():
    print(json.dumps(CFG.dict(), indent=2))
