import importlib
import operator
import sys
import warnings
from typing import Optional

from packaging import version

from superduperdb.base.exceptions import RequiredPackageNotFound

ops = {
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
    "<": operator.lt,
}


def _compare_versions(op, got_ver, want_ver, requirement, pkg):
    if not ops[op](version.parse(got_ver), version.parse(want_ver)):
        raise RequiredPackageNotFound(
            f"{requirement} is required for a normal"
            f"functioning of this module, but found {pkg}=={got_ver}"
        )


def check_versioned(module_name, pkg, got_ver):
    if len(module_name) == 3:
        op = None
        if module_name[1] is None:
            # Upper bounded version check
            want_ver = module_name[-1]
            op = '<='
            _compare_versions(op, got_ver, want_ver, f'{pkg}{op}{want_ver}', pkg)
        elif module_name[-1] is None:
            # Lower bounded version check
            want_ver = module_name[1]
            op = '>='
            _compare_versions(op, got_ver, want_ver, f'{pkg}{op}{want_ver}', pkg)
        else:
            # Lower and upper bounded version check
            want_ver_lower = module_name[1]
            want_ver_upper = module_name[-1]
            _compare_versions(
                '>=', got_ver, want_ver_lower, f'{pkg}>={want_ver_lower}', pkg
            )
            _compare_versions(
                '<=', got_ver, want_ver_upper, f'{pkg}<={want_ver_upper}', pkg
            )

    elif len(module_name) == 2:
        # Exact version check
        got_ver = importlib.metadata.version(module_name[0])
        want_ver = module_name[1]
        _compare_versions('==', got_ver, want_ver, f'{pkg}<={want_ver}', pkg)

    else:
        raise ValueError(
            f'Cannot check the package requirement for the module {module_name}'
        )


def requires_packages(*modules):
    '''
    A utility function to check required packages for a module
    i.e superduperdb.ext
    '''
    missing_modules = []
    missing_module_versions = []

    def _create_versioned(module):
        if len(module) == 2:
            return '=='.join(module)
        elif module[1] is None:
            return f'{module[0]}<={module[-1]}'
        elif module[-1] is None:
            return f'{module[0]}>={module[1]}'
        else:
            return f'{module[0]}>={module[1]},<={module[-1]}'

    for module_name in modules:
        pkg = module_name[0]
        if len(module_name) == 1:
            try:
                importlib.import_module(module_name[0])
            except ImportError:
                missing_modules.append(module_name[0])
        else:
            got_ver = None
            try:
                got_ver = importlib.metadata.version(module_name[0])
                check_versioned(module_name, pkg, got_ver)
            except RequiredPackageNotFound:
                missing_module_versions.append((module_name, pkg, got_ver))
            except ImportError:
                missing_modules.append(module_name[0])

    missing_module_versions += missing_modules

    if missing_module_versions:
        msg = []
        reqs = []
        for v in missing_module_versions:
            if isinstance(v, str):
                reqs.append(v)
                msg.append(f'Module: {v} not installed')
            else:
                reqs.append(_create_versioned(v[0]))
                msg.append(f'Module: {v[1]} Required: {v[0]} but got {v[1]}=={v[-1]}')
        reqs = '\n'.join(reqs)

        raise RequiredPackageNotFound(
            f"The following modules are required but "
            "either not present or installed with wrong "
            f"version: {','.join(msg)}"
            "\nAdd following lines to requirements.txt:"
            f"\n{reqs}"
        )


def public_api(stability: str = 'stable'):
    """Annotation for documenting public APIs.

    If ``stability="alpha"``, the API can be used by advanced users who are
    tolerant to and expect breaking changes.

    If ``stability="beta"``, the API is still public and can be used by early
    users, but are subject to change.

    If ``stability="stable"``, the APIs will remain backwards compatible across
    minor releases.
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
    """Specialized Deprecation Warning for fine grained filtering control"""

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
