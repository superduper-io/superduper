import os
from pathlib import Path

from superduper import Template

PARENT = Path(__file__).resolve().parent.parent.parent


def ls():
    return [x.split('.')[0] for x in os.listdir(PARENT / "templates") if str(x).endswith("zip")] 


def __getattr__(name: str):
    return Template.read(str(PARENT / "templates" / f"{name}"))
