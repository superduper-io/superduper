"""
Operations on dictionaries used to fill and combine config files
and environment variables
"""

import os
import sys
import typing as t
from pathlib import Path

import fil

Dict = t.Dict[str, object]
Files = t.Sequence[t.Union[Path, str]]
StrDict = t.Dict[str, str]

SEP = '_'
_NONE = object()


def config_dicts(files: Files, parent: StrDict, prefix: str, environ: StrDict) -> Dict:
    data = _read_all(files)
    environ_dict = _environ_to_config_dict(prefix, parent, environ)
    return _combine((*data, environ_dict))


def _read_all(files: Files, fail: bool = False) -> t.Sequence[Dict]:
    if fail:
        return [fil.read(f) for f in files]
    else:
        return [fil.read(f, {}) for f in files]


def _combine(dicts: t.Sequence[Dict]) -> Dict:
    result: Dict = {}
    for d in dicts:
        _combine_one(result, d)
    return result


def _environ_to_config_dict(
    prefix: str,
    parent: StrDict,
    environ: t.Optional[StrDict] = None,
    err: t.Optional[t.TextIO] = sys.stderr,
    fail: bool = False,
):
    env_dict = _environ_dict(prefix, environ)

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


def _split_address(
    key: str, parent: StrDict
) -> t.Iterator[t.Tuple[StrDict, t.Tuple[str]]]:
    def split(key, parent, *address):
        if key in parent:
            yield *address, key

        for k, v in parent.items():
            if key.startswith(ks := k + SEP) and isinstance(v, dict):
                yield from split(key[len(ks) :], v, *address, k)

    return split(key, parent)


def _environ_dict(prefix: str, environ: t.Optional[StrDict] = None) -> StrDict:
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
    env_dict: StrDict, parent: StrDict
) -> t.Tuple[StrDict, StrDict]:
    good: t.Dict = {}
    bad: t.Dict = {}

    for k, v in env_dict.items():
        addresses = list(_split_address(k, parent))
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
