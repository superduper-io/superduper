import os
from pathlib import Path

from superduper import Template

PARENT = Path(__file__).resolve().parent

BASE_URL = 'https://superduper-public-templates.s3.us-east-2.amazonaws.com'

VERSIONS = {
    'llm_finetuning': '0.4.0',
    'multimodal_image_search': '0.4.0',
    'multimodal_video_search': '0.4.0',
    'pdf_rag': '0.4.0',
    'rag': '0.4.0',
    'simple_rag': '0.4.0',
    'text_vector_search': '0.4.0',
    'transfer_learning': '0.4.0',
}


TEMPLATES = {k: BASE_URL + f'/{k}-{VERSIONS[k]}.zip' for k in VERSIONS}


def ls():
    """List all available templates."""
    return TEMPLATES


def __getattr__(name: str):
    assert name in TEMPLATES
    file = TEMPLATES[name].split('/')[-1]
    if not os.path.exists(f'/tmp/{file}'):
        import subprocess

        subprocess.run(['curl', '-O', '-k', TEMPLATES[name]])
        subprocess.run(['mv', file, f'/tmp/{file}'])
        subprocess.run(['unzip', f'/tmp/{file}', '-d', f'/tmp/{file[:-4]}'])
    t = Template.read(f'/tmp/{file[:-4]}')
    requirements_path = f'{file[:-4]}/requirements.txt'
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            t.requirements = [x.strip() for x in f.read().split('\n') if x]
    return t
