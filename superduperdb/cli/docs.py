import functools
import os

from typer import Option

from superduperdb import ROOT
from superduperdb.misc import run

from . import command

DOCS = 'apidocs'
DOCS_ROOT = ROOT / DOCS
SOURCE_ROOT = DOCS_ROOT / 'source'
CODE_ROOT = ROOT / 'superduperdb'
TMP_DIR = ROOT / '.cache/apidocs'

run_cmd = functools.partial(run.run, cwd=TMP_DIR)


@command(help='Build documentation')
def docs(
    _open: bool = Option(
        False,
        '-o',
        '--open',
        help='If true, open the index.html of the generated pages on completion',
    ),
):
    os.makedirs(TMP_DIR, exist_ok=True)
    run_cmd(('sphinx-apidoc', '-f', '-o', str(SOURCE_ROOT), str(CODE_ROOT)))
    run_cmd(('sphinx-build', '-a', str(DOCS_ROOT), '.'))
    if _open:
        run_cmd(('open', 'index.html'))
