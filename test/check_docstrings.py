import ast
import os


def extract_classes_functions_and_docstrings(directory):
    """
    Extract and print class names, their docstrings, public method names
    with docstrings, and standalone function names with docstrings
    from all .py files in the given directory.

    :param directory: The directory path to search for Python files.
    """
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.py'):
                file_path = os.path.join(subdir, filename)
                with open(file_path, 'r') as file:
                    file_content = file.read()
                    try:
                        ast_tree = ast.parse(file_content)
                        for node in ast.walk(ast_tree):
                            if isinstance(
                                node, ast.ClassDef
                            ) and not node.name.startswith('_'):
                                class_name = node.name
                                class_docstring = (
                                    ast.get_docstring(node) or "No docstring"
                                )
                                print(
                                    f"Class: {class_name}\nDocstring: {class_docstring}"
                                )
                                for element in node.body:
                                    if isinstance(
                                        element, ast.FunctionDef
                                    ) and not element.name.startswith('_'):
                                        method_name = element.name
                                        method_docstring = (
                                            ast.get_docstring(element) or "No docstring"
                                        )
                                        print(
                                            f"\tMethod: {method_name}"
                                            f"\n\tDocstring: {method_docstring}"
                                        )
                                print("\n")
                            elif isinstance(
                                node, ast.FunctionDef
                            ) and not node.name.startswith('_'):
                                function_name = node.name
                                function_docstring = (
                                    ast.get_docstring(node) or "No docstring"
                                )
                                print(
                                    f"Function: {function_name}\n"
                                    f"Docstring: {function_docstring}\n"
                                )
                    except SyntaxError as e:
                        print(f"Syntax error in file {file_path}: {e}")


# Example usage:
directory_path = './superduperdb'
extract_classes_functions_and_docstrings(directory_path)
