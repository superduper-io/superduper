import subprocess
import typing as t
from subprocess import PIPE, CalledProcessError

import superduperdb as s

__all__ = (
    'CalledProcessError',
    'PIPE',
    'run',
    'out',
)


def run(
    args: t.Sequence[str], text: bool = True, check: bool = True, **kwargs
) -> subprocess.CompletedProcess:
    if s.CFG.verbose:
        print('$', *args)
    return subprocess.run(args, text=text, check=check, **kwargs)


def out(args: t.Sequence[str], **kwargs) -> str:
    return run(args, stdout=PIPE, **kwargs).stdout.strip()
