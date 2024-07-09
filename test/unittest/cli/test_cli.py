import json

from superduper.misc import run


def test_cli_info():
    data = run.out(('python', '-m', 'superduper', 'info')).strip()
    assert data.startswith('```') and data.endswith('```')
    json.loads(data[3:-3])
