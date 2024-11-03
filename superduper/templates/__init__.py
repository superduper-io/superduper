import os
from pathlib import Path

from superduper import Template

PARENT = Path(__file__).resolve().parent


def ls():
    """List all available templates."""
    return [
        x.split('.')[0]
        for x in os.listdir(PARENT)
        if not x.startswith('.') and not x.startswith('__')
    ]


def __getattr__(name: str):
    t = Template.read(str(PARENT / f"{name}"))
    requirements_path = str(PARENT / f"{name}" / "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            t.requirements = [x.strip() for x in f.read().split('\n') if x]
    return t
