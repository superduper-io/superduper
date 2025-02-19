# TODO - test adding variables to identifier
import re


def _find_variables(r):
    from superduper.base.base import Base

    if isinstance(r, dict):
        return sum([_find_variables(v) for v in r.values()], [])
    if isinstance(r, (list, tuple)):
        return sum([_find_variables(v) for v in r], [])
    if isinstance(r, str):
        return re.findall(r'<var:(.*?)>', r)
    if isinstance(r, Base):
        return r.variables
    return []


def _replace_variables(x, **kwargs):
    from .document import Document

    if isinstance(x, dict):
        return {
            _replace_variables(k, **kwargs): _replace_variables(v, **kwargs)
            for k, v in x.items()
        }
    if (
        isinstance(x, str)
        and re.match(r'^<var:(.*?)>$', x) is not None
        and '<' not in x[1:-1]
    ):
        return kwargs.get(x[5:-1], x)
    if isinstance(x, str):
        variables = re.findall(r'<var:(.*?)>', x)
        if variables:
            print(x)
        variables = list(map(lambda v: v.strip(), variables))
        for k in variables:
            if k not in kwargs:
                continue
            x = x.replace(f'<var:{k}>', str(kwargs[k]))
        return x
    if isinstance(x, (list, tuple)):
        return [_replace_variables(v, **kwargs) for v in x]
    if isinstance(x, Document):
        return x.set_variables(**kwargs)
    return x
