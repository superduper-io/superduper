from subprocess import CalledProcessError, PIPE
from typing import Any, Dict, Tuple
import subprocess

__all__ = (
    'CalledProcessError',
    'PIPE',
    'run',
    'out',
)


def run(
    args: Tuple[str], text: bool = True, check: bool = True, **kwargs: Dict[str, Any]
) -> subprocess.CompletedProcess:
    print('$', args)
    return subprocess.run(args, text=text, check=check, **kwargs)


def out(args: Tuple[str], **kwargs: Dict[str, Any]) -> str:
    return run(args, stdout=PIPE, **kwargs).stdout.strip()
