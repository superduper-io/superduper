# Scrape data from the inline documentation of a Python package

Install requirements. [Sphinx](https://sphinx-rtd-tutorial.readthedocs.io/) required to convert doc-strings to Markdown documents, which can then be easily 
parsed.


```python
!pip install superduperdb
!pip install sphinx
!pip install sphinx-markdown-builder
```

Convert the `pymongo` inline documentation to markdown files.


```bash
%%bash
git clone git@github.com:mongodb/mongo-python-driver.git
cd mongo-python-driver
echo 'extensions.append("sphinx_markdown_builder")' >> doc/conf.py
sphinx-apidoc -f -o source pymongo/
mkdir output
sphinx-build -a -b markdown doc output
```

Parse those files.


```python
import os
import re

PARENT_DIR = './mongo-python-driver/output/api/pymongo'

documents = os.listdir(f'{PARENT_DIR}/')
data = []

for file in documents:
    with open(f'{PARENT_DIR}/{file}') as f:
        content = f.read()
    split = re.split(r'^(#{1,4}) ', content, flags=re.MULTILINE)
    split = [(split[2 * i - 1], split[2 * i]) for i in range(1, len(split) // 2)]
    last_key = None
    for item in split:
        type_ = item[0]
        content = item[1]
        key = content.split('\n')[0]
        key = re.split('[:\(\*]', re.sub('\*[a-z]+\*', '', key).strip())[0]
        value = '\n'.join(content.split('\n')[1:])
        info = {}
        if type_ in {'###', '####'}:
            if type_ == '###':
                if last_key is None:
                    last_key = key
                info['key'] = key
                info['parent'] = None
                last_key = key
            elif type_ == '####':
                info['parent'] = last_key
                info['key'] = key
            info['value'] = value[:120]
            info['document'] = file
            if ' ' in key:
                continue
            if re.match('^[A-Z]{2,}$', key):
                continue
            if 'Version' in key:
                continue
            info['res'] = key
            data.append(info)
```

Save the documentation as JSON.


```python
import json
with open('pymongo.json', 'w') as f:
    json.dump(data, f)
```
