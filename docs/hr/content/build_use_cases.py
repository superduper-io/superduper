import json
import os
import typing as t
import re

from fastapi import FastAPI
import uvicorn


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
            for cell in snippet_nb['cells']:
                if '<testing:' in '\n'.join(cell['source']):
                    continue
                built_nb['cells'].append(cell)
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

def build_notebook_from_tabs(path, selected_tabs):
    with open(path) as f:
        nb = json.load(f)

    built_nb = {k: v for k, v in nb.items() if k != 'cells'}
    built_nb['cells'] = []
    ix = 0

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
            snippet_tab = selected_tabs[ix]
            ix += 1

            with open(f'docs/reusable_snippets/{snippet}.ipynb') as f:
                snippet_nb = json.load(f)
            snippet_nb = process_snippet(snippet_nb, tabs)
            snippet_tab_cell = None

            for cell in snippet_nb['cells']:
                if '<testing:' in '\n'.join(cell['source']):
                    continue
                if '<tab:' in '\n'.join(cell['source']):
                    matches = re.findall(r'<tab: (.+?)>', cell['source'][0])
                    if not snippet_tab_cell:
                        if snippet_tab == "":
                            built_nb['cells'].append(cell)
                        else:
                            if matches[0] == snippet_tab:
                                built_nb['cells'].append(cell)
                        snippet_tab_cell = True

                else:
                    built_nb['cells'].append(cell)

    exported_path = f'./built-notebook-{os.path.basename(path)}'
    with open(exported_path, 'w') as f:
        json.dump(built_nb, f)
    return exported_path

def serve_notebook_builder():
    app = FastAPI()

    @app.post("/build_notebook")
    def build(usecase_path: str, tabs: t.List[str]):
        return build_notebook_from_tabs(usecase_path, tabs)

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == '__main__':
    serve_notebook_builder()
