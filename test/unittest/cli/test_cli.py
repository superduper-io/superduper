import json

from superduperdb.misc import run


def test_cli_info():
    data = run.out(('python', '-m', 'superduperdb', 'info'))
    json.loads(data)
