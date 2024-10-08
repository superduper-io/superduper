#!/usr/bin/env python3
"""
Script to generate or update README.md files for Python plugin projects.

This script reads information from pyproject.toml and source code files to
generate a standardized README.md for each plugin in the plugins directory.
"""

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import toml

README_TEMPLATE = """# {{plugin_name}}

{{description}}

## Installation

```bash
pip install {{plugin_name}}
```

## API


- [Code](https://github.com/superduper-io/superduper/tree/main/plugins/{{name}})
- [API-docs](/docs/api/plugins/{{plugin_name}})

| Class | Description |
|---|---|
{{classes_table}}

{{examples}}"""


def extract_pyproject_info(pyproject_path: Path) -> Dict[str, str]:
    """
    Extract necessary information from pyproject.toml.

    Args:
    ----
        pyproject_path (Path): Path to the pyproject.toml file.

    Returns:
    -------
        Dict[str, str]: A dictionary containing the extracted information.

    """
    with pyproject_path.open("r", encoding="utf-8") as f:
        pyproject_data = toml.load(f)
    project = pyproject_data.get("project", {})
    name = project.get("name", "")
    description = project.get("description", "")
    urls = project.get("urls", {})
    source_url = urls.get("source", "")
    return {
        "name": name,
        "description": description,
        "source_url": source_url,
        "api_docs_url": f"/docs/api/plugins/{name}",
    }


def parse_example(docstring: str) -> Optional[List[Dict[str, str]]]:
    """
    Parse the docstring to extract an example with text descriptions and code blocks.

    Args:
    ----
        docstring (str): The docstring of a class or method.

    Returns:
    -------
        Optional[List[Dict[str, str]]]: A list of example parts, each part is a
                                        dict with 'type' and 'content'.

    """
    lines = docstring.split('\n')
    in_example_section = False
    example_parts = []
    current_part = {'type': None, 'content': []}

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line.startswith('Example:'):
            in_example_section = True
            continue
        if in_example_section:
            if stripped_line.strip() == '-------':
                continue
            if not stripped_line:
                # Empty line
                if current_part['content']:
                    # Finish the current part
                    example_parts.append(
                        {
                            'type': current_part['type'],
                            'content': '\n'.join(current_part['content']).strip(),
                        }
                    )
                    current_part = {'type': None, 'content': []}
                continue
            if stripped_line.startswith('>>>'):
                # Code line
                if current_part['type'] != 'code':
                    if current_part['content']:
                        # Finish the current part
                        example_parts.append(
                            {
                                'type': current_part['type'],
                                'content': '\n'.join(current_part['content']).strip(),
                            }
                        )
                    current_part = {'type': 'code', 'content': []}
                code_line = line.strip()[4:]
                current_part['content'].append(code_line)
            else:
                # Text line
                if current_part['type'] != 'text':
                    if current_part['content']:
                        # Finish the current part
                        example_parts.append(
                            {
                                'type': current_part['type'],
                                'content': '\n'.join(current_part['content']).strip(),
                            }
                        )
                    current_part = {'type': 'text', 'content': []}
                current_part['content'].append(stripped_line)
    # Add the last part if any
    if current_part['content']:
        example_parts.append(
            {
                'type': current_part['type'],
                'content': '\n'.join(current_part['content']).strip(),
            }
        )

    return example_parts if example_parts else None


def get_classes_with_docstrings(
    package_dir: Path, package_name: str
) -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
    """
    Parse Python files to extract classes and their docstrings.

    Args:
    ----
        package_dir (Path): Path to the package directory containing Python modules.
        package_name (str): The package's importable name.

    Returns:
    -------
        List[Dict[str, Union[str, List[Dict[str, str]]]]]: A list of dictionaries
                                                           with class information.

    """
    classes_info = []
    for py_file in package_dir.rglob("*.py"):
        with py_file.open("r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read(), filename=str(py_file))
            except SyntaxError:
                continue  # Skip files with syntax errors
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                if class_name.startswith("_"):
                    continue
                module_path = py_file.relative_to(package_dir.parent)
                module_name = ".".join(module_path.with_suffix("").parts)
                full_class_name = f"{module_name}.{class_name}"
                docstring = ast.get_docstring(node)
                if docstring:
                    summary_line = docstring.strip().split("\n")[0]
                    if '# noqa' in docstring:
                        continue
                    example_parts = parse_example(docstring)
                else:
                    summary_line = ""
                    example_parts = None
                    continue
                classes_info.append(
                    {
                        "class_name": f"{class_name}",
                        "full_class_name": f"`{full_class_name}`",
                        "description": summary_line,
                        "example_parts": example_parts,
                    }
                )
    return classes_info


def generate_readme_content(
    info: Dict[str, str],
    classes_info: List[Dict[str, Union[str, List[Dict[str, str]]]]],
) -> str:
    """
    Generate the README.md content based on extracted information.

    :param info: Extracted information from pyproject.toml.
    :param classes_info: Extracted classes and their docstrings.
    """
    plugin_name = info["name"]
    # Replace plugin name in the template
    template = README_TEMPLATE.replace("{{plugin_name}}", plugin_name)

    # Replace name in the source url
    template = template.replace(
        "{{name}}",
        plugin_name.split("_", 1)[1],
    )

    # Replace the description
    template = template.replace("{{description}}", info["description"])

    # Generate the classes table
    classes_table = ""
    for cls in classes_info:
        class_name = cls["full_class_name"]
        description = cls["description"]
        classes_table += f"| {class_name} | {description} |\n"
    template = template.replace("{{classes_table}}", classes_table)

    # Generate the examples section
    examples = ""
    for cls in classes_info:
        example_parts = cls.get("example_parts")
        if example_parts:
            examples += f"### {cls['class_name']}\n\n"
            for part in example_parts:
                if part['type'] == 'text':
                    examples += part['content'] + "\n\n"
                elif part['type'] == 'code':
                    examples += "```python\n"
                    examples += part['content'] + "\n"
                    examples += "```\n\n"

    if examples:
        examples = "## Examples\n\n" + examples
    template = template.replace("{{examples}}", examples)
    return template


def update_readme(readme_path: Path, new_content: str) -> None:
    """
    Update the README.md file with new content, preserving existing sections.

    Args:
    ----
        readme_path (Path): Path to the README.md file.
        new_content (str): The new content to insert into the README.md.

    """
    auto_gen_start = "<!-- Auto-generated content start -->\n"
    auto_gen_end = "<!-- Auto-generated content end -->\n"
    if readme_path.exists():
        with readme_path.open("r", encoding="utf-8") as f:
            existing_content = f.read()
        # Replace the auto-generated content
        pattern = re.compile(
            re.escape(auto_gen_start) + ".*?" + re.escape(auto_gen_end), re.DOTALL
        )
        if pattern.search(existing_content):
            new_full_content = pattern.sub(
                auto_gen_start + new_content + "\n" + auto_gen_end, existing_content
            )
        else:
            # If markers not found, prepend the new content
            new_full_content = (
                auto_gen_start
                + new_content
                + "\n"
                + auto_gen_end
                + "\n"
                + existing_content
            )
    else:
        # Create a new README.md with placeholders
        new_full_content = auto_gen_start + new_content + "\n" + auto_gen_end
        new_full_content += "\n<!-- Add your additional content below -->\n"
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(new_full_content)


def process_plugin(plugin_dir: Path) -> None:
    """
    Process a single plugin directory to generate or update its README.md.

    Args:
    ----
        plugin_dir (Path): Path to the plugin directory.

    """
    print(f"Processing plugin at {plugin_dir}")
    pyproject_path = plugin_dir / "pyproject.toml"
    if not pyproject_path.exists():
        print(f"pyproject.toml not found in {plugin_dir}, skipping.")
        return
    info = extract_pyproject_info(pyproject_path)
    # Find the package directory
    package_name = info["name"].replace("-", "_")
    package_dir = plugin_dir / package_name
    if not package_dir.exists():
        print(f"Package directory {package_dir} not found, skipping.")
        return
    classes_info = get_classes_with_docstrings(package_dir, package_name)
    new_readme_content = generate_readme_content(info, classes_info)
    readme_path = plugin_dir / "README.md"
    update_readme(readme_path, new_readme_content)
    print(f"Updated README.md at {readme_path}")


def main():
    """Main function to process all plugins in the plugins directory."""
    parser = argparse.ArgumentParser(
        description="Generate or update README.md files for plugins."
    )
    parser.add_argument(
        "plugin_path",
        nargs="?",
        default=None,
        help="Path to the plugin directory (e.g., plugins/openai)",
    )
    args = parser.parse_args()

    if args.plugin_path:
        plugin_dir = Path(args.plugin_path)
        if not plugin_dir.is_dir():
            print(f"Plugin directory {plugin_dir} does not exist.")
            return
        process_plugin(plugin_dir)
    else:
        plugins_dir = Path("plugins")
        for plugin_path in plugins_dir.iterdir():
            if plugin_path.is_dir():
                if plugin_path.name in {"dummy", "template"}:
                    continue
                process_plugin(plugin_path)


if __name__ == "__main__":
    main()
