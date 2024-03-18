import json
import os
import re


def process_snippet(nb, tabs):
    to_delete = []
    for i, cell in enumerate(nb['cells']):
        if tabs != '*':
            match = re.match('^#[ ]+<tab: ([^>]+)>', cell['source'][0])
            if match:
                tab = match.groups()[0]
                if tab not in tabs:
                    to_delete.append(i)
                    continue
        if cell['cell_type'] == 'markdown':
            for j, line in enumerate(cell['source']):
                line = re.sub('^####', '##### ', line)
                line = re.sub('^###', '#### ', line)
                line = re.sub('^## ', '### ', line)
                line = re.sub('^# ', '## ', line)
                nb['cells'][i]['source'][j] = line
    nb['cells'] = [cell for i, cell in enumerate(nb['cells']) if i not in to_delete]
    return nb


def build_use_case(path):
    with open(path) as f:
        nb = json.load(f)
    built_nb = {k: v for k, v in nb.items() if k != 'cells'}
    built_nb['cells'] = []

    for cell in nb['cells']:
        if (
            cell['cell_type'] == 'raw' 
            and cell['source'] 
            and cell['source'][0].startswith('<snippet:')
        ):
            snippet, tabs = re.match(
                '^<snippet: ([a-z0-9_\-]+): ([a-zA-Z0-9_\-\,\*]+)>$',
                cell['source'][0].strip(),
            ).groups()
            with open(f'docs/reusable_snippets/{snippet}.ipynb') as f:
                snippet_nb = json.load(f)
            snippet_nb = process_snippet(snippet_nb, tabs)
            built_nb['cells'].extend(snippet_nb['cells'])
        else:
            built_nb['cells'].append(cell)
    return built_nb

file = '_multimodal_vector_search.ipynb'


for file in os.listdir('./use_cases'):
    if not file.startswith('_'):
        continue
    built = build_use_case(f'./use_cases/{file}')
    with open(f'./use_cases/{file[1:]}', 'w') as f:
        json.dump(built, f)
