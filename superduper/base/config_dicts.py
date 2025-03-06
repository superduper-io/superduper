"""Utility functions for combining and converting dictionaries.

Operations on dictionaries used to fill and combine config files
and environment variables
"""

import os
import sys
import typing as t

Dict = t.Dict[str, object]
StrDict = t.Dict[str, str]

SEP = '_'
_NONE = object()


def combine_configs(dicts: t.Sequence[Dict]) -> Dict:
    """Combine a sequence of dictionaries into a single dictionary.

    :param dicts: The dictionaries to combine.
    """
    result: Dict = {}
    for d in dicts:
        _combine_one(result, d)
    return result


def environ_to_config_dict(
    prefix: str,
    parent: StrDict,
    environ: t.Optional[StrDict] = None,
    err: t.Optional[t.TextIO] = sys.stderr,
    fail: bool = False,
):
    """Convert environment variables to a configuration dictionary.

    :param prefix: The prefix to use for environment variables.
    :param parent: The parent dictionary to use as a basis.
    :param environ: The environment variables to read from.
    :param err: The file to write errors to.
    :param fail: Whether to raise an exception on error.
    :return: The configuration dictionary.
    """
    env_dict = _environ_dict(prefix, environ)
    good, bad = _env_dict_to_config_dict(env_dict, parent)
    bad = {k: v for k, v in bad.items() if k != 'SUPERDUPER_CONFIG'}
    try:
        bad['unknown'] = list(set(bad['unknown']) - {'config'})
        if not bad['unknown']:
            del bad['unknown']
    except KeyError:
        pass

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

        if isinstance(old_v, bool):
            v = _fix_bool(v)

        if old_v is _NONE:
            target[k] = v

        elif (
            not (isinstance(v, type(old_v)) or isinstance(old_v, type(v)))
            and old_v is not None
            and v is not None
        ):
            err = f'Expected {type(old_v)} but got {type(v)} for {k}={v}'
            raise ValueError(err)

        elif isinstance(v, dict):
            _combine_one(old_v, v)

        else:
            target[k] = v


def _fix_bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ('true', '1'):
        v = True
    elif v.lower() in ('false', '0'):
        v = False

    return v


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
