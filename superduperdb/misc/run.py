from subprocess import CalledProcessError, PIPE
import typing as t
import subprocess

__all__ = (
    'CalledProcessError',
    'PIPE',
    'run',
    'out',
)


def run(
    args: t.Tuple[str], text: bool = True, check: bool = True, **kwargs: t.Any
) -> subprocess.CompletedProcess:
    print('$', *args)
    return subprocess.run(args, text=text, check=check, **kwargs)


def out(args: t.Tuple[str], **kwargs: t.Any) -> str:
    return run(args, stdout=PIPE, **kwargs).stdout.strip()
