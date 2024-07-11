import os
import sys

from setuptools import setup

if not os.getenv("IGNORE_ERROR"):
    error_message = """
    ERROR: This package has been renamed to 'superduper'. Please install the new package using:
    
    pip install superduper

    Github: https://github.com/superduper-io/superduper
    """
    print(error_message)
    sys.exit(1)

setup(
    name="superduperdb",
    version="0.2.2",
    description="This package has been deprecated. Use superduper instead.",
    packages=[],
)