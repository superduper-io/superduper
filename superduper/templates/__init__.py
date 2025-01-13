import os
from pathlib import Path

from superduper import Template

PARENT = Path(__file__).resolve().parent

BASE_URL = 'https://superduper-public-templates.s3.us-east-2.amazonaws.com'

VERSIONS = {
    'llm_finetuning': '0.5.0',
    'multimodal_image_search': '0.5.0',
    'multimodal_video_search': '0.5.0',
    'pdf_rag': '0.5.0',
    'rag': '0.5.0',
    'simple_rag': '0.5.0',
    'text_vector_search': '0.5.0',
    'transfer_learning': '0.5.0',
}


TEMPLATES = {k: BASE_URL + f'/{k}-{VERSIONS[k]}.zip' for k in VERSIONS}


def ls():
    """List all available templates."""
    return TEMPLATES


def __getattr__(name: str):
    import re

    if not re.match('.*[0-9]+\.[0-9]+\.[0-9]+.*', name):
        assert name in TEMPLATES, f'{name} not in supported templates {TEMPLATES}'
        file = TEMPLATES[name].split('/')[-1]
        url = TEMPLATES[name]
    else:
        file = name + '.zip'
        url = BASE_URL + f'/{name}.zip'

    if not os.path.exists(f'/tmp/{file}'):
        import subprocess

        subprocess.run(['curl', '-O', '-k', url])
        subprocess.run(['mv', file, f'/tmp/{file}'])
        subprocess.run(['unzip', f'/tmp/{file}', '-d', f'/tmp/{file[:-4]}'])

    t = Template.read(f'/tmp/{file[:-4]}')
    requirements_path = f'{file[:-4]}/requirements.txt'
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            t.requirements = [x.strip() for x in f.read().split('\n') if x]
    return t


def get(name):
    """Get a template from local cache or online hub.

    :param name: name of template
    """
    return __getattr__(name)
