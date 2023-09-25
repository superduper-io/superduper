import json

from superduperdb.misc import run


def test_cli_info():
    data = run.out(('python', '-m', 'superduperdb', 'info')).strip()
    assert data.startswith('```') and data.endswith('```')
    json.loads(data[3:-3])
