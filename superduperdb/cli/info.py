import datetime
import json
import os
import platform
import socket
import sys
from pathlib import Path

from superduperdb import ROOT
from superduperdb.misc import run

from . import command

PYPROJECT = ROOT / 'pyproject.toml'


@command(help='Print information about the current machine and installation')
def info():
    print('```')
    print(json.dumps(_get_info(), default=str, indent=2))
    print('```')


def _get_info():
    return {
        'cfg': _cfg(),
        'cwd': Path.cwd(),
        'freeze': _freeze(),
        'git': _git(),
        'hostname': socket.gethostname(),
        'os_uname': list(os.uname()),
        'package_versions': _package_versions(),
        'platform': _platform(),
        'startup_time': datetime.datetime.now(),
        'superduper_db_root': ROOT,
        'sys': _sys(),
    }


def _cfg():
    try:
        from superduperdb import CFG

        return CFG.dict()
    except Exception:
        return '(CFG not yet commited)'


def _freeze():
    try:
        from pip._internal.operations.freeze import freeze

        return list(freeze())
    except Exception as e:
        return [f'Freeze failed with {e}']


def _git():
    def run_out(*cmd):
        try:
            return run.out(cmd, cwd=ROOT)
        except Exception as e:
            return f'{cmd} failed with {e}'

    return {
        'branch': run_out('git', 'branch', '--show-current'),
        'commit': run_out('git', 'show', '-s', '--format="%h: %s"'),
    }


def _package_versions():
    return {}


def _platform():
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
    }


def _sys():
    return {'argv': sys.argv, 'path': sys.path}
