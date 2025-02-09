import sys
import warnings
from collections import defaultdict


class SuperDuperDeprecationWarning(DeprecationWarning):
    """
    Specialized Deprecation Warning for fine grained filtering control.

    :param args: *args of `DeprecationWarning`
    :param kwargs: **kwargs of `DeprecationWarning`
    """


if not sys.warnoptions:
    warnings.filterwarnings("module", category=SuperDuperDeprecationWarning)


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
