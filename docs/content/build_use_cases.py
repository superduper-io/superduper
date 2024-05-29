import json
import os
import sys
import typing as t
import re

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
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
            with open(f'reusable_snippets/{snippet}.ipynb') as f:
                snippet_nb = json.load(f)
            snippet_nb = process_snippet(snippet_nb, tabs)
            for cell in snippet_nb['cells']:
                if '<testing:' in '\n'.join(cell['source']):
                    continue
                built_nb['cells'].append(cell)
        else:
            built_nb['cells'].append(cell)
    return built_nb


def build_use_cases():
    for file in os.listdir('./use_cases'):
        if not file.startswith('_'):
            continue
        print(file)
        built = build_use_case(f'./use_cases/{file}')
        with open(f'./use_cases/{file[1:]}', 'w') as f:
            json.dump(built, f)
        

def get_snippet(snippet_nb_cells, snippet_tab):

    snippet_cells = []
    snippet_tab_cell = None
    for cell in snippet_nb_cells:
        if '<testing:' in '\n'.join(cell['source']):
            continue
        if '<tab:' in '\n'.join(cell['source']):
            matches = re.findall(r'<tab: (.+?)>', cell['source'][0])
            if not snippet_tab_cell:
                if snippet_tab == "":
                    snippet_cells.append(cell)
                    snippet_tab_cell = True
                else:
                    if matches[0] == snippet_tab:
                        snippet_cells.append(cell)
                        snippet_tab_cell = True
        else:
            snippet_cells.append(cell)
    return snippet_cells


def build_notebook_from_tabs(path, selected_tabs):
    with open(path) as f:
        nb = json.load(f)

    built_nb = {k: v for k, v in nb.items() if k != 'cells'}
    built_nb['cells'] = []
    ix = 0
    snippets_group = []
    non_snippet_group = []

    for cell in nb['cells']:
        if (
            cell['cell_type'] == 'raw' 
            and cell['source'] 
            and cell['source'][0].startswith('<snippet:')
        ):
            
            if non_snippet_group:
                snippets_group.append(non_snippet_group)
                non_snippet_group = []

            snippet, tabs = re.match(
                '^<snippet: ([a-z0-9_\-]+): ([a-zA-Z0-9_\-\,\*]+)>$',
                cell['source'][0].strip(),
            ).groups()
            with open(f'reusable_snippets/{snippet}.ipynb') as f:
                snippet_nb = json.load(f)
            snippet_nb = process_snippet(snippet_nb, tabs)
            if tabs != '*':
                if len(tabs.split(',')) == 1:
                    continue
            snippets_group.append(snippet_nb['cells'])

        elif cell['cell_type'] == 'code' and '<tab:' in '\n'.join(cell['source']):
            non_snippet_group.append(cell)

        elif cell['cell_type'] == 'markdown':
            if non_snippet_group:
                snippets_group.append(non_snippet_group)

            snippets_group.append((cell, ))
            non_snippet_group = []
        else:
            if non_snippet_group:
                snippets_group.append(non_snippet_group)
                non_snippet_group = []
            snippets_group.append((cell, ))

    if non_snippet_group:
        snippets_group.append(non_snippet_group)
        non_snippet_group = []
    for cell in snippets_group:
        if isinstance(cell, tuple):
            built_nb['cells'].append(cell[0])
        else:
            cells = get_snippet(cell, snippet_tab=selected_tabs[ix])
            for cell in cells:
                built_nb['cells'].append(cell)
            ix += 1


    notebook_bytes  = json.dumps(built_nb).encode('utf-8')
    return Response(content=notebook_bytes, media_type="application/octet-stream")


def serve_notebook_builder():
    app = FastAPI()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )
    @app.post("/build_notebook")
    def build(usecase_path: str, tabs: t.List[str]):
        return build_notebook_from_tabs(usecase_path, tabs)

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == '__main__':
    if not sys.argv[1:] or sys.argv[1] == 'build':
        build_use_cases()
    elif sys.argv[1] == 'serve':
        serve_notebook_builder()
