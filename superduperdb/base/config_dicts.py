"""
Operations on dictionaries used to fill and combine config files
and environment variables
"""
from __future__ import annotations

import os
import sys
import typing as t
from pathlib import Path

import fil

SEP = '_'
_NONE = object()


def read_all(files: t.Sequence[Path | str], fail: bool = False) -> t.Sequence[dict]:
    if fail:
        return [fil.read(f) for f in files]
    else:
        return [fil.read(f, {}) for f in files]


def combine(dicts: t.Sequence[dict]) -> dict:
    result: dict = {}
    for d in dicts:
        _combine_one(result, d)
    return result


def environ_to_config_dict(
    prefix: str,
    parent: dict,
    environ: dict | None = None,
    err: t.TextIO | None = sys.stderr,
    fail: bool = False,
):
    env_dict = environ_dict(prefix, environ)

    good, bad = _env_dict_to_config_dict(env_dict, parent)

    if bad:
        bad = {k: ', '.join([prefix + i.upper() for i in v]) for k, v in bad.items()}
        msg = '\n'.join(f'{k}: {v}' for k, v in sorted(bad.items()))

        s = 's' * (sum(len(v) for v in bad.values()) != 1)
        msg = f'Bad environment variable{s}:\n{msg}'
        if err is not None:
            print(msg, file=err)
        if fail:
            raise ValueError(msg)

    return good


def split_address(key: str, parent: dict) -> t.Iterator[tuple[dict, tuple[str]]]:
    def split(key, parent, *address):
        if key in parent:
            yield *address, key

        for k, v in parent.items():
            if key.startswith(ks := k + SEP) and isinstance(v, dict):
                yield from split(key[len(ks) :], v, *address, k)

    return split(key, parent)


def environ_dict(prefix: str, environ: dict | None = None) -> dict:
    if not (prefix.isupper() and prefix.endswith(SEP) and not prefix.startswith(SEP)):
        raise ValueError(f'Bad prefix={prefix}')

    d = os.environ if environ is None else environ
    items = ((k, v) for k, v in d.items() if k.isupper() and k.startswith(prefix))
    return {k[len(prefix) :].lower(): v for k, v in items}


def _combine_one(target, source):
    for k, v in source.items():
        old_v = target.get(k, _NONE)
        if old_v is _NONE:
            target[k] = v

        elif not (isinstance(v, type(old_v)) or isinstance(old_v, type(v))):
            err = f'Expected {type(old_v)} but got {type(v)} for key={k}'
            raise ValueError(err)

        elif isinstance(v, dict):
            _combine_one(old_v, v)

        else:
            target[k] = v


def _env_dict_to_config_dict(
    env_dict: dict[str, str], parent: dict
) -> tuple[dict, dict]:
    good: dict = {}
    bad: dict = {}

    for k, v in env_dict.items():
        addresses = list(split_address(k, parent))
        if not addresses:
            bad.setdefault('unknown', []).append(k)
        elif len(addresses) > 1:
            bad.setdefault('ambiguous', []).append(k)
        else:
            d = good
            *address, last = addresses[0]
            for a in address:
                d = d.setdefault(a, {})
            d[last] = v

    return good, bad
