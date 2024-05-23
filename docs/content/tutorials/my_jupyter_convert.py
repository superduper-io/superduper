import base64
import json
import os
import sys

notebook = sys.argv[1]

with open(notebook, "r") as f:
    notebook = json.load(f)

print(json.dumps(notebook, indent=2))

text = ''

FILE_NAME = sys.argv[1].split('.')[0]
OUTPUT_DIRECTORY = '../../static/' + FILE_NAME

os.system('mkdir -p ' + OUTPUT_DIRECTORY)

for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        text += "\n```python\n" + ''.join(cell['source']) + '\n```\n'

    if 'outputs' in cell:
        blocks = []
        for j, output in enumerate(cell['outputs']):
            if 'text' in output:
                tmp = ''.join(['    ' + x for x in output['text']])
                tmp = tmp.replace('<', '\<').replace('>', '\>')
                tmp = tmp.replace('{', '\{').replace('}', '\}')
                blocks.append('<pre>\n' + tmp + '\n</pre>')
            if 'data' in output:
                if 'image/png' in output['data']:
                    with open(OUTPUT_DIRECTORY + f'/{i}_{j}.png', 'wb') as f:
                        f.write(base64.b64decode(output['data']['image/png']))
                    blocks.append(f'<div>![](/{FILE_NAME}/{i}_{j}.png)</div>')
                    continue
                if 'text/plain' in output['data']:
                    tmp = ''.join(['    ' + x for x in output['data']['text/plain']])
                    tmp = tmp.replace('<', '\<').replace('>', '\>')
                    tmp = tmp.replace('{', '\{').replace('}', '\}')
                    blocks.append('<pre>\n' + tmp + '\n</pre>')

        blocks = '\n'.join(blocks)
        text += "\n<details>\n<summary>Outputs</summary>\n" + blocks + '\n</details>\n'

    if cell['cell_type'] == 'markdown':
        text += '\n' + ''.join(cell['source']) + '\n'

with open(sys.argv[1].replace('.ipynb', '.md'), "w") as f:
    f.write(text)