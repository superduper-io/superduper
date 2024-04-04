import ast
import os
import re

import pytest


class _BaseDocstringException(Exception):
    def __init__(self, file_path, node, *args, parent=None, msg: str = ''):
        super().__init__(*args)
        self.filename = file_path
        self.node = node
        self.msg = msg
        self.parent = parent

    def __str__(self) -> str:
        if self.parent:
            return (
                f'{self.msg} in {self.filename}:'
                f'{self.node.lineno} - {self.parent}.{self.node.name}'
            )
        else:
            return (
                f'{self.msg} in {self.filename}:{self.node.lineno} - {self.node.name}'
            )


class MissingDocstring(_BaseDocstringException):
    def __init__(self, file_path, node, *args, parent=None):
        super().__init__(
            file_path=file_path,
            node=node,
            msg='Found no docstring',
            parent=parent,
        )


class MismatchingDocParameters(_BaseDocstringException):
    ...


class MissingParameterExplanation(_BaseDocstringException):
    ...


def get_class_init_params(node):
    init_params = []
    for node in node.body:
        if isinstance(node, ast.FunctionDef) and node.name == '__init__':
            for arg in node.args.args:
                if arg.arg != 'self':
                    init_params.append(arg.arg)
            break
    return init_params


def get_function_params(node):
    params = []
    for arg in node.args.args:
        params.append(arg.arg)
    return params


def get_method_params(node):
    params = []
    for arg in node.args.args:
        if arg.arg != 'self':
            params.append(arg.arg)
    return params


def get_dataclass_init_params(node):
    init_params = []
    for item in node.body:
        if isinstance(item, ast.AnnAssign):
            annotation = ast.unparse(item.annotation)
            if 'ClassVar' not in annotation:
                field_name = item.target.id
                init_params.append(field_name)
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    field_name = target.id
                    init_params.append(field_name)
    init_params = [p for p in init_params if p != '__doc__']
    return init_params


def get_doc_string_params(doc_string):
    param_pattern = r":param (\w+): (.+)"
    params = re.findall(param_pattern, doc_string)
    return {param[0]: param[1] for param in params}


def check_class_docstring(file_path, node, dataclass=False):
    print(f'{file_path}::{node.name}')
    doc_string = ast.get_docstring(node)
    if doc_string is None:
        raise MissingDocstring(file_path, node)

    if dataclass:
        params = get_dataclass_init_params(node)
    else:
        params = get_class_init_params(node)
    doc_params = get_doc_string_params(doc_string)

    if len(doc_params) != len(params):
        raise MismatchingDocParameters(
            file_path=file_path,
            node=node,
            msg=f'Got {len(params)} parameters but doc-string has {len(doc_params)}.',
        )

    for i, (p, (dp, expl)) in enumerate(zip(params, doc_params.items())):
        if p != dp:
            raise MismatchingDocParameters(
                file_path=file_path, node=node, msg=f'At position {i}: {p} != {dp}'
            )
        if not expl.strip():
            raise MissingParameterExplanation(
                file_path=file_path,
                node=node,
                msg=f'Missing explanation of parameter {dp}',
            )


def check_method_docstring(file_path, parent_class, node):
    print(f'{file_path}::{parent_class}::{node.name}')
    doc_string = ast.get_docstring(node)
    if doc_string is None:
        raise MissingDocstring(file_path, node, parent=parent_class)

    params = get_method_params(node)
    doc_params = get_doc_string_params(doc_string)

    if len(doc_params) != len(params):
        raise MismatchingDocParameters(
            file_path=file_path,
            node=node,
            msg=f'Got {len(params)} parameters but doc-string has {len(doc_params)}.',
            parent=parent_class,
        )

    for i, (p, (dp, expl)) in enumerate(zip(params, doc_params.items())):
        if p != dp:
            raise MismatchingDocParameters(
                file_path=file_path,
                node=node,
                msg=f'At position {i}: {p} != {dp}',
                parent=parent_class,
            )
        if not expl.strip():
            raise MissingParameterExplanation(
                file_path=file_path,
                node=node,
                msg=f'Missing explanation of parameter {dp}',
                parent=parent_class,
            )


def check_function_doc_string(file_path, node):
    print(f'{file_path}::{node.name}')
    doc_string = ast.get_docstring(node)
    if doc_string is None:
        raise MissingDocstring(file_path, node)

    params = get_function_params(node)
    doc_params = get_doc_string_params(doc_string)

    if len(doc_params) != len(params):
        raise MismatchingDocParameters(
            file_path=file_path,
            node=node,
            msg=f'Got {len(params)} parameters but doc-string has {len(doc_params)}.',
        )

    for i, (p, (dp, expl)) in enumerate(zip(params, doc_params.items())):
        if p != dp:
            raise MismatchingDocParameters(
                file_path=file_path,
                node=node,
                msg=f'At position {i}: {p} != {dp}',
            )
        if not expl.strip():
            raise MissingParameterExplanation(
                file_path=file_path,
                node=node,
                msg=f'Missing explanation of parameter {dp}',
            )


def is_dataclass(node):
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call):
            if (
                isinstance(decorator.func, ast.Name)
                and decorator.func.id == 'dataclass'
            ) or (
                isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == 'dataclass'
            ):
                return True
        elif isinstance(decorator, ast.Name) and decorator.id == 'dataclass':
            return True
        elif isinstance(decorator, ast.Attribute) and decorator.attr == 'dataclass':
            return True
    return False


def extract_docstrings(directory):
    class_test_cases = []
    method_test_cases = []
    function_test_cases = []

    for subdir, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.py'):
                file_path = os.path.join(subdir, filename)
                with open(file_path, 'r') as file:
                    file_content = file.read()
                    try:
                        ast_tree = ast.parse(file_content)
                        for node in ast.iter_child_nodes(ast_tree):
                            if isinstance(
                                node, ast.ClassDef
                            ) and not node.name.startswith('_'):
                                class_test_cases.append((file_path, node))
                                for item in node.body:
                                    if isinstance(
                                        item, ast.FunctionDef
                                    ) and not item.name.startswith('_'):
                                        skip = False
                                        for decorator in item.decorator_list:
                                            if (
                                                isinstance(decorator, ast.Name)
                                                and decorator.id == 'override'
                                            ):
                                                skip = True
                                        if not skip:
                                            method_test_cases.append(
                                                (file_path, node.name, item)
                                            )
                            elif isinstance(
                                node, ast.FunctionDef
                            ) and not node.name.startswith('_'):
                                function_test_cases.append((file_path, node))
                    except SyntaxError as e:
                        print(f"Syntax error in file {file_path}: {e}")
    return class_test_cases, method_test_cases, function_test_cases


CLASS_TEST_CASES, METHOD_TEST_CASES, FUNCTION_TEST_CASES = extract_docstrings(
    './superduperdb'
)


print(f'Found {len(CLASS_TEST_CASES)} class __init__ documentation test-cases')
print(f'Found {len(METHOD_TEST_CASES)} method documentation test-cases')
print(f'Found {len(FUNCTION_TEST_CASES)} function documentation test-cases')


@pytest.mark.skip
@pytest.mark.parametrize("test_case", CLASS_TEST_CASES)
def test_class_docstrings(test_case):
    file_path, node = test_case
    check_class_docstring(file_path=file_path, node=node, dataclass=is_dataclass(node))


@pytest.mark.skip
@pytest.mark.parametrize("test_case", METHOD_TEST_CASES)
def test_method_docstrings(test_case):
    file_path, node = test_case
    check_function_doc_string(
        file_path=file_path,
        node=node,
    )


@pytest.mark.skip
@pytest.mark.parametrize("test_case", FUNCTION_TEST_CASES)
def test_function_docstrings(test_case):
    file_path, node = test_case
    check_function_doc_string(
        file_path=file_path,
        node=node,
    )
