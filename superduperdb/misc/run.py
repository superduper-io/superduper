import subprocess
import typing as t
from subprocess import PIPE, CalledProcessError

__all__ = (
    'CalledProcessError',
    'PIPE',
    'run',
    'out',
)


def run(
    args: t.Tuple[str], text: bool = True, check: bool = True, **kwargs
) -> subprocess.CompletedProcess:
    print('$', *args)
    return subprocess.run(args, text=text, check=check, **kwargs)


def out(args: t.Tuple[str], **kwargs) -> str:
    return run(args, stdout=PIPE, **kwargs).stdout.strip()
