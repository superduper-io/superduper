import datetime
import importlib
import json
import platform
import socket
import sys
import typing as t
from pathlib import Path

from superduperdb import ROOT
from superduperdb.base.exceptions import RequiredPackageVersionsNotFound

from . import command

PYPROJECT = ROOT / 'pyproject.toml'


@command(help='Print information about the current machine and installation')
def info():
    """Print information about the current machine and installation."""
    print('```')
    print(json.dumps(_get_info(), default=str, indent=2))
    print('```')


@command(help='Print information about the current machine and installation')
def requirements(ext: t.List[str]):
    """Print information about the current machine and installation.

    :param ext: Extensions to check.
    """
    out = []
    for e in ext:
        try:
            m = importlib.import_module(f'superduperdb.ext.{e}')
            out.extend(m.requirements)
        except RequiredPackageVersionsNotFound as e:
            out.extend([x for x in str(e).split('\n') if x])
    print('\n'.join(out))


def _get_info():
    return {
        'cfg': _cfg(),
        'cwd': Path.cwd(),
        'freeze': _freeze(),
        'hostname': socket.gethostname(),
        'os_uname': list(platform.uname()),
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


def _package_versions():
    return {}


def _platform():
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
    }


def _sys():
    return {'argv': sys.argv, 'path': sys.path}
