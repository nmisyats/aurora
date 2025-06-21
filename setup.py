import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "aurora",
    version = "0.0.1",
    author = "Nazar Misyats",
    packages=['aurora'],
    entry_points={
        'console_scripts': [
            'aurora = aurora.cli.app:main',
        ]
    }
)