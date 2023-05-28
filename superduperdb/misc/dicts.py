from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, TextIO, Tuple, Union
import fil
import os
import sys

SEP = '_'
_NONE = object()


def read_all(files: Sequence[Union[Path, str]], fail=False) -> Sequence[Dict]:
    if fail:
        return [fil.read(f) for f in files]
    else:
        return [fil.read(f, {}) for f in files]


def combine(dicts: Sequence[Dict]) -> Dict:
    result = {}
    for d in dicts:
        _combine_one(result, d)
    return result


def environ_to_config_dict(
    prefix: str,
    parent: Dict,
    environ: Optional[Dict] = None,
    err: Optional[TextIO] = sys.stderr,
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


def split_address(key: str, parent: Dict) -> Iterator[Tuple[Dict, Tuple[str]]]:
    def split(key, parent, *address):
        if key in parent:
            yield *address, key

        for k, v in parent.items():
            if key.startswith(ks := k + SEP) and isinstance(v, dict):
                yield from split(key[len(ks) :], v, *address, k)

    return split(key, parent)


def environ_dict(prefix: str, environ: Optional[Dict] = None) -> Dict:
    assert prefix.isupper() and prefix.endswith(SEP) and not prefix.startswith(SEP)

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


def _env_dict_to_config_dict(env_dict: Dict, parent: Dict) -> Dict:
    good, bad = {}, {}

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
