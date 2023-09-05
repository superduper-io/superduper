import logging
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
    args: t.Sequence[str],
    text: bool = True,
    check: bool = True,
    verbose: bool = False,
    **kwargs,
) -> subprocess.CompletedProcess:
    """
    Run a command, printing it if verbose is enabled.

    :param args: The command to run.
    :param text: Whether to use text mode.
    :param check: Whether to raise an error if the command fails.
    :param **kwargs: Additional arguments to pass to ``subprocess.run``.
    """
    if verbose:
        logging.info('$', *args)
    return subprocess.run(args, text=text, check=check, **kwargs)


def out(args: t.Sequence[str], **kwargs) -> str:
    """
    Run a command and return the output.

    :param args: The command to run.
    :param **kwargs: Additional arguments to pass to ``subprocess.run``.
    """
    return run(args, stdout=PIPE, **kwargs).stdout.strip()
