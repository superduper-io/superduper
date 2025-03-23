import inspect
import re

import pytest


class _BaseDocstringException(Exception):
    def __init__(
        self,
        module,
        name,
        *args,
        msg: str = '',
        parent=None,
        line=None,
    ):
        super().__init__(*args)
        self.module = module
        self.name = name
        self.msg = msg
        self.parent = parent
        self.line = line

    def __str__(self):
        return (
            f'{self.msg} in {self.module}.{self.name} of {self.parent}\n'
            f'path: {self.module.replace(".", "/")}.py:{self.line}'
        )


class MissingDocstring(_BaseDocstringException):
    def __init__(
        self,
        module,
        name,
        *args,
        parent=None,
        line=None,
    ):
        super().__init__(
            module=module,
            name=name,
            msg='Found no docstring',
            parent=parent,
            line=line,
        )


class MismatchingDocParameters(_BaseDocstringException): ...


class MissingParameterExplanation(_BaseDocstringException): ...


def get_doc_string_params(doc_string):
    param_pattern = r":param (\w+): (.+)"
    params = re.findall(param_pattern, doc_string)
    params = [param for param in params if not param[0].startswith('*')]
    return {param[0]: param[1] for param in params}


def check_class_docstring(cls, line):
    print(f'{cls.__module__}.{cls.__name__}')
    doc_string = cls.__doc__
    if doc_string is None:
        raise MissingDocstring(cls.__module__, cls.__name__)

    params = [k for k in inspect.signature(cls.__init__).parameters if k != 'self']
    doc_params = get_doc_string_params(doc_string)
    doc_params_set = set(doc_params.keys())
    if doc_params_set != set(params):
        diff = (set(params) - set(doc_params.keys())).union(
            set(doc_params.keys()) - set(params)
        )
        raise MismatchingDocParameters(
            module=cls.__module__,
            name=cls.__name__,
            msg=(
                f'Got {len(params)} parameters but doc-string has {len(doc_params)}.\n'
                f'{params}\nvs.\n{list(doc_params.keys())}\n'
                f'diff is {diff}'
            ),
            line=line,
        )

    for i, (p, (dp, expl)) in enumerate(zip(params, doc_params.items())):
        if not expl.strip():
            raise MissingParameterExplanation(
                module=cls.__module__,
                name=cls.__name__,
                msg=f'Missing explanation of parameter {dp}',
                line=line,
            )


def check_method_docstring(method, line):
    doc_string = method.__doc__
    if doc_string is None:
        msg = str(method)
        raise MissingDocstring(method.__module__, msg, line=line)

    params = {
        k: v for k, v in inspect.signature(method).parameters.items() if k != 'self'
    }
    doc_params = get_doc_string_params(doc_string)

    if len(doc_params) != len(params):
        raise MismatchingDocParameters(
            module=method.__module__,
            name=str(method),
            msg=f'Got {len(params)} parameters but doc-string has {len(doc_params)}.',
            parent=None,
            line=line,
        )

    for i, (p, (dp, expl)) in enumerate(zip(params, doc_params.items())):
        if p != dp:
            raise MismatchingDocParameters(
                module=method.__module__,
                name=str(method),
                msg=f'At position {i}: {p} != {dp}',
                parent=None,
                line=line,
            )
        if not expl.strip():
            raise MissingParameterExplanation(
                module=method.__module__,
                name=str(method),
                msg=f'Missing explanation of parameter {dp}',
                parent=None,
                line=line,
            )


def check_function_doc_string(function, line):
    print(f'{function.__module__}.{function.__name__}')
    doc_string = function.__doc__
    if doc_string is None:
        raise MissingDocstring(function.__module__, function.__name__)

    params = inspect.signature(function).parameters
    doc_params = get_doc_string_params(doc_string)

    if len(doc_params) != len(params):
        raise MismatchingDocParameters(
            module=function.__module__,
            name=function.__name__,
            msg=f'Got {len(params)} parameters but doc-string has {len(doc_params)}.',
            line=line,
        )

    for i, (p, (dp, expl)) in enumerate(zip(params, doc_params.items())):
        if p != dp:
            raise MismatchingDocParameters(
                module=function.__module__,
                name=function.__name__,
                msg=f'At position {i}: {p} != {dp}',
                line=line,
            )
        if not expl.strip():
            raise MissingParameterExplanation(
                module=function.__module__,
                name=function.__name__,
                msg=f'Missing explanation of parameter {dp}',
                line=line,
            )


def list_all_members(package, prefix=None, seen=None):
    if seen is None:
        seen = set()
    if prefix is None:
        prefix = package.__name__

    members = []

    for name, obj in inspect.getmembers(package):
        if (
            inspect.ismodule(obj)
            and obj.__name__.startswith(prefix)
            and obj.__name__ not in seen
        ):
            seen.add(obj.__name__)
            members.extend(list_all_members(obj, prefix, seen))
        elif inspect.isfunction(obj):
            if obj.__module__.startswith(prefix):
                members.append(
                    (obj, obj.__module__, name, None, 'function', obj.__doc__)
                )
        elif inspect.isclass(obj):
            if obj.__module__.startswith(prefix):
                members.append((obj, obj.__module__, name, None, 'class', obj.__doc__))
                # Inspect methods within the class
                class_methods = inspect.getmembers(obj, predicate=inspect.isfunction)
                for meth_name, meth_obj in class_methods:
                    if meth_obj.__module__ == obj.__module__:
                        members.append(
                            (
                                meth_obj,
                                meth_obj.__module__,
                                name,
                                meth_name,
                                'method',
                                meth_obj.__doc__,
                            )
                        )

    return members


def extract_docstrings():
    import superduper

    members = list_all_members(package=superduper)
    # for subpackage in os.listdir('superduper/ext'):
    #     if subpackage.startswith('_') or subpackage == 'utils.py':
    #         continue
    #     exec(f'import superduper.ext.{subpackage}')
    #     package = eval(f'superduper.ext.{subpackage}')
    #     tmp = list_all_members(package=package, prefix=f'superduper.ext.{subpackage}')
    #     members.extend(tmp)
    from superduper.misc.special_dicts import DeepKeyedDict

    lookup = DeepKeyedDict({})

    for m, module, item, child, type, doc in members:
        try:
            line = inspect.getsourcelines(m)[1]
        except OSError:
            line = None
        if child is None:
            lookup[f'{module}.{item}'] = {
                '::type': type,
                '::item': item,
                '::doc': doc,
                '::object': m,
                '::line': line,
            }
            continue
        lookup[f'{module}.{item}.{child}'] = {
            '::type': type,
            '::item': item,
            '::doc': doc,
            '::object': m,
            '::line': line,
        }
    return lookup


TEST_CASES = extract_docstrings()
FUNCTION_TEST_CASES = []
CLASS_TEST_CASES = []
METHOD_TEST_CASES = []

for k in TEST_CASES.keys(True):
    parent = k.split('.::')[0]
    v = TEST_CASES[parent]
    if isinstance(v['::doc'], str) and 'noqa' in v['::doc']:
        continue

    if (
        v['::type'] == 'method'
        and not v['::item'].startswith('_')
        and not parent.split('.')[-1].startswith('_')
    ):
        METHOD_TEST_CASES.append(parent)

    if v['::type'] == 'class' and not parent.split('.')[-1].startswith('_'):
        CLASS_TEST_CASES.append(parent)

    if v['::type'] == 'function' and not parent.split('.')[-1].startswith('_'):
        FUNCTION_TEST_CASES.append(parent)


CLASS_TEST_CASES = {
    '/'.join(k.split('.')[:-1]) + f'.py:{TEST_CASES[k]["::line"]}': k
    for k in CLASS_TEST_CASES
}

FUNCTION_TEST_CASES = {
    '/'.join(k.split('.')[:-1]) + f'.py:{TEST_CASES[k]["::line"]}': k
    for k in FUNCTION_TEST_CASES
}

METHOD_TEST_CASES = {
    '/'.join(k.split('.')[:-2]) + f'.py:{TEST_CASES[k]["::line"]}': k
    for k in METHOD_TEST_CASES
}


CLASS_TEST_CASES_KEYS = sorted(list(set(CLASS_TEST_CASES.keys())))
METHOD_TEST_CASES_KEYS = sorted(list(set(METHOD_TEST_CASES.keys())))
FUNCTION_TEST_CASES_KEYS = sorted(list(set(FUNCTION_TEST_CASES.keys())))

print(f'Found {len(CLASS_TEST_CASES)} class __init__ documentation test-cases')
print(f'Found {len(FUNCTION_TEST_CASES)} function documentation test-cases')
print(f'Found {len(METHOD_TEST_CASES)} method documentation test-cases')


@pytest.mark.parametrize("test_case", CLASS_TEST_CASES_KEYS)
def test_class_docstrings(test_case):
    test_case = CLASS_TEST_CASES[test_case]
    check_class_docstring(
        TEST_CASES[test_case]["::object"],
        TEST_CASES[test_case]["::line"],
    )


@pytest.mark.parametrize("test_case", FUNCTION_TEST_CASES_KEYS)
def test_function_docstrings(test_case):
    test_case = FUNCTION_TEST_CASES[test_case]
    check_function_doc_string(
        TEST_CASES[test_case]["::object"],
        TEST_CASES[test_case]["::line"],
    )


@pytest.mark.parametrize("test_case", METHOD_TEST_CASES_KEYS)
def test_method_docstrings(test_case):
    test_case = METHOD_TEST_CASES[test_case]
    check_method_docstring(
        TEST_CASES[test_case]["::object"],
        TEST_CASES[test_case]["::line"],
    )
