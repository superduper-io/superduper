import functools
import importlib
import sys
import typing as t
import warnings
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

    :param *packages: list of tuples of packages
                     each tuple of the form
                     (import_name, lower_bound/None,
                      upper_bound/None, install_name/None)
    :param warn: if True, warn instead of raising an exception

    E.g. ('sklearn', '0.1.0', '0.2.0', 'scikit-learn')
    """
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


# TODO: add deprecated also
def public_api(stability: str = 'stable'):
    """Annotation for documenting public APIs.

    If ``stability="alpha"``, the API can be used by advanced users who are
    tolerant to and expect breaking changes.

    If ``stability="beta"``, the API is still public and can be used by early
    users, but are subject to change.

    If ``stability="stable"``, the APIs will remain backwards compatible across
    minor releases.

    :param stability: stability of the API
    """
    assert stability in ["stable", "beta", "alpha"]

    def wrap(obj):
        if stability in ["alpha", "beta"]:
            message = (
                f"**public_api({stability}):** This API is in {stability} "
                "and may change before becoming stable."
            )
            _append_doc(obj, message=message)
        return obj

    return wrap


class SuperDuperDBDeprecationWarning(DeprecationWarning):
    """Specialized Deprecation Warning for fine grained filtering control."""

    pass


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


def ui(*schema: t.Dict, handle_integration: t.Callable = lambda x: x):
    """Annotation for documenting UI schemas.

    :param *schema: list of dictionaries representing the UI schema
    :param handle_integration: function to handle the integration of the UI schema
    """

    def decorated(f):
        f.get_ui_schema = lambda: schema
        f.build = lambda r: f(**r)
        f.handle_integration = handle_integration
        return f

    return decorated
