from collections import defaultdict
import os
import re
import shutil
import sys

sys.path.insert(0, os.getcwd())

from superduper.misc.annotations import extract_parameters, replace_parameters
from test.unittest.test_docstrings import (
    FUNCTION_TEST_CASES,
    CLASS_TEST_CASES,
    TEST_CASES,
)

lookup = defaultdict(list)

for v in FUNCTION_TEST_CASES.values():
    parent = '.'.join(v.split('.')[:-1])
    lookup[parent].append(TEST_CASES[v])

for v in CLASS_TEST_CASES.values():
    parent = '.'.join(v.split('.')[:-1])
    lookup[parent].append(TEST_CASES[v])

shutil.rmtree('docs/hr/content/api', ignore_errors=True)
os.makedirs('docs/hr/content/api', exist_ok=True)

import re


def format_docstring(docstring):
    """
    Formats a docstring by creating a markdown table for parameters,
    converting interactive Python shell code blocks to markdown code blocks,
    and organizing the rest of the text into separate sections.

    :param docstring: The original docstring to format.
    """
    docstring = '\n'.join([line.strip() for line in docstring.strip().split('\n')])
    # Remove and build parameter table
    params = extract_parameters(docstring)
    docstring = replace_parameters(docstring, '')

    # Build markdown table for parameters if any
    markdown_table = ''
    if params:
        markdown_table = "| Parameter | Description |\n|-----------|-------------|\n"
        for name, desc in params.items():
            desc = ' '.join(desc)
            markdown_table += f"| {name} | {desc.strip()} |\n"
        markdown_table += "\n"

    # Identify and format code blocks
    # Split the docstring on two or more newlines to identify separate blocks
    blocks = re.split(r'\n{2,}', docstring)
    formatted_blocks = []
    for block in blocks:
        if '>>>' in block:
            # Format as a code block
            formatted_block = '```python\n'
            lines = block.split('\n')
            for line in lines:
                if line.startswith('>>>'):
                    # Remove '>>>' and strip spaces
                    formatted_block += line[4:].strip() + '\n'
                elif line.startswith('...'):
                    # Append continuation lines directly
                    formatted_block += '    ' + line[4:].strip() + '\n'
                else:
                    # Handle outputs separately, assumed to be outputs if not starting with '>>>'
                    formatted_block += '# ' + line.strip() + '\n'
            formatted_block += '```'
            formatted_blocks.append(formatted_block)
        else:
            # Normal text blocks
            formatted_blocks.append(block.strip())

    # Combine all parts into the formatted content
    formatted_content = '\n\n'.join([block for block in formatted_blocks if block])
    if markdown_table:
        formatted_content = markdown_table + formatted_content
    return formatted_content.strip()


for k in lookup:
    content = f'**`{k}`** \n\n'
    content += f"[Source code](https://github.com/superduper/superduper/blob/main/{k.replace('.', '/')}.py)\n\n"
    for node in lookup[k]:
        if node['::item'].startswith('_'):
            continue
        ds = node['::doc']
        if ds is None:
            continue
        content += f"## `{node['::item']}` \n\n"

        import inspect

        if inspect.isclass(node['::object']):
            sig = str(inspect.signature(node['::object'].__init__))
        else:
            sig = str(inspect.signature(node['::object']))

        sig = ',\n    '.join(sig.split(','))
        content += "```python\n" + node['::item'] + sig + '\n```\n'
        ds = format_docstring(ds)
        content += ds + '\n\n'

    k = k.replace('.', '/')
    child = k.split('/')[-1]
    parent = '/'.join(k.split('/')[1:-1])
    os.makedirs('docs/content/api/' + parent, exist_ok=True)
    with open(f'docs/content/api/{parent}/{child}.md', 'w') as f:
        f.write(content)
