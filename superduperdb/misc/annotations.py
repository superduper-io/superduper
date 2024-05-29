import functools
import hashlib
import importlib
import inspect
import sys
import typing as t
import warnings
from collections import defaultdict
from importlib import metadata
from typing import Optional

from packaging import version

from superduperdb import logging
from superduperdb.base.exceptions import RequiredPackageVersionsNotFound


def _normalize_module(import_module, lower_bound, upper_bound, install_module):
    assert import_module is not None
    if install_module is None:
        install_module = import_module
    if upper_bound is None:
        upper_bound = f'{sys.maxsize}.0.0'
    if lower_bound is None:
        lower_bound = '0.0.0'
    if install_module is None:
        install_module = import_module
    return (
        import_module,
        version.parse(lower_bound),
        version.parse(upper_bound),
        install_module,
    )


MIN = version.parse('0.0.0')
MAX = version.parse(f'{sys.maxsize}.0.0')


def _compare_versions(package, lower_bound, upper_bound, install_name):
    constraint = ''
    if lower_bound == upper_bound:
        constraint = f'=={lower_bound}'
    elif lower_bound > MIN and upper_bound < MAX:
        constraint = f'>={lower_bound},<={upper_bound}'
    elif upper_bound < MAX:
        constraint = f'<={upper_bound}'
    elif lower_bound > MIN:
        constraint = f'>={lower_bound}'
    installation_line = f'{install_name}{constraint}'
    try:
        got_version = version.parse(metadata.version(package))
    except metadata.PackageNotFoundError:
        try:
            got_version = version.parse(importlib.import_module(package).__version__)
        except (metadata.PackageNotFoundError, ModuleNotFoundError):
            logging.error(f'Could not find package {package}')
            return False, installation_line + '    # (no such package installed)'
    if not (lower_bound <= got_version and got_version <= upper_bound):
        return False, installation_line + f'    # (got {got_version})'
    return True, installation_line


def requires_packages(*packages, warn=False):
    """Require the packages to be installed.

    :param packages: list of tuples of packages
                     each tuple of the form
                     (import_name, lower_bound/None,
                      upper_bound/None, install_name/None)
    :param warn: if True, warn instead of raising an exception

    E.g. ('sklearn', '0.1.0', '0.2.0', 'scikit-learn')
    """
    from superduperdb import REQUIRES

    out = []
    all = []
    for m in packages:
        satisfactory, install_line = _requires_packages(*m)
        if not satisfactory:
            out.append(install_line)
        all.append(install_line)
    if out:
        if warn:
            warnings.warn('\n' + '\n'.join(out))
        else:
            raise RequiredPackageVersionsNotFound('\n' + '\n'.join(out))
    REQUIRES.extend(all)
    return out, all


def _requires_packages(
    import_module, lower_bound=None, upper_bound=None, install_module=None
):
    """Compare the versions of the required packages.

    A utility function to check that a required package for a module
    in superduperdb.ext is installed.
    """
    import_module, lower_bound, upper_bound, install_module = _normalize_module(
        import_module,
        lower_bound,
        upper_bound,
        install_module,
    )
    return _compare_versions(import_module, lower_bound, upper_bound, install_module)


def deprecated(f):
    """Decorator to mark a function as deprecated.

    This will result in a warning being emitted when the function is used.

    :param f: function to deprecate
    """

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        logging.warn(
            f"{f.__name__} is deprecated and will be removed in a future release.",
        )
        return f(*args, **kwargs)

    return decorated


class SuperDuperDBDeprecationWarning(DeprecationWarning):
    """
    Specialized Deprecation Warning for fine grained filtering control.

    :param args: *args of `DeprecationWarning`
    :param kwargs: **kwargs of `DeprecationWarning`
    """


if not sys.warnoptions:
    warnings.filterwarnings("module", category=SuperDuperDBDeprecationWarning)


def _append_doc(obj, *, message: str, directive: Optional[str] = None):
    if not obj.__doc__:
        obj.__doc__ = ""

    obj.__doc__ = obj.__doc__.rstrip()

    indent = _get_indent(obj.__doc__)
    obj.__doc__ += "\n\n"

    if directive is not None:
        obj.__doc__ += f"{' ' * indent}.. {directive}::\n\n"

        message = message.replace("\n", "\n" + " " * (indent + 4))
        obj.__doc__ += f"{' ' * (indent + 4)}{message}"
    else:
        message = message.replace("\n", "\n" + " " * (indent + 4))
        obj.__doc__ += f"{' ' * indent}{message}"
    obj.__doc__ += f"\n{' ' * indent}"


def _get_indent(docstring: str) -> int:
    if not docstring:
        return 0

    non_empty_lines = list(filter(bool, docstring.splitlines()))
    if len(non_empty_lines) == 1:
        return 0

    return len(non_empty_lines[1]) - len(non_empty_lines[1].lstrip())


def component(*schema: t.Dict):
    """Decorator for creating a component.

    :param schema: schema for the component
    """

    def decorator(f):
        @functools.wraps(f)
        def decorated(*, db=None, **kwargs):
            import inspect

            if 'db' in inspect.signature(f).parameters:
                out = f(**kwargs, db=db)
            else:
                out = f(**kwargs)

            from superduperdb.components.component import Component

            assert isinstance(out, Component)

            def _deep_flat_encode(cache, blobs, files, leaves_to_keep=(), schema=None):
                h = hashlib.sha1(str(kwargs).encode()).hexdigest()
                path = f'{f.__module__}.{f.__name__}'.replace('.', '/')
                id = f'{path}/{h}'
                cache[id] = {'_path': path, **kwargs}
                return f'?{id}'

            out._deep_flat_encode = _deep_flat_encode
            return out

        return decorated

    return decorator


def merge_docstrings(cls):
    """Decorator that merges Sphinx-styled class docstrings.

    Decorator merges doc-strings from parent to child classes,
    ensuring no duplicate parameters and correct indentation.

    :param cls: Class to merge docstrings for.
    """
    parent_doc = next(
        (parent.__doc__ for parent in inspect.getmro(cls)[1:] if parent.__doc__), None
    )
    if parent_doc:
        parent_params = extract_parameters(parent_doc)
        child_doc = cls.__doc__ or """"""
        child_params = extract_parameters(child_doc)
        for k in child_params:
            parent_params[k] = child_params[k]
        placeholder_doc = replace_parameters(child_doc)
        param_string = ''
        for k, v in parent_params.items():
            v = '\n    '.join(v)
            param_string += f':param {k}: {v}\n'
        cls.__doc__ = placeholder_doc.replace('!!!', param_string)
    return cls


def extract_parameters(doc):
    """
    Extracts and organizes parameter descriptions from a Sphinx-styled docstring.

    :param doc: Sphinx-styled docstring.
                Docstring may have multiple lines
    """
    lines = [x.strip() for x in doc.split('\n')]
    was_doc = False
    import re

    params = defaultdict(list)
    for line in lines:
        if line.startswith(':param'):
            was_doc = True
            match = re.search(r':param[ ]+(.*):(.*)$', line)
            param = match.groups()[0]
            params[param].append(match.groups()[1].strip())
        if not line.startswith(':') and was_doc and line.strip():
            params[param].append(line.strip())
        if not line.strip():
            was_doc = False
    return params


def replace_parameters(doc, placeholder: str = '!!!'):
    """
    Replace parameters in a doc-string with a placeholder.

    :param doc: Sphinx-styled docstring.
    :param placeholder: Placeholder to replace parameters with.
    """
    doc = [x.strip() for x in doc.split('\n')]
    lines = []
    had_parameters = False
    parameters_done = False
    for line in doc:
        if parameters_done:
            lines.append(line)
            continue

        if not had_parameters and line.startswith(':param'):
            lines.append(placeholder)
            had_parameters = True
            assert not parameters_done, 'Can\'t have multiple parameter sections'
            continue

        if had_parameters and line.startswith(':param'):
            continue

        if not line.strip() and had_parameters:
            parameters_done = True

        if had_parameters and not parameters_done:
            continue

        lines.append(line)

    if not had_parameters:
        lines = lines + ['\n' + placeholder]

    return '\n'.join(lines)


if __name__ == '__main__':
    print(replace_parameters(extract_parameters.__doc__))
