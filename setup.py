import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "aurora",
    version = "1.0.0",
    author = "Nazar Misyats",
    packages=['aurora', 'aurora_cli'],
    entry_points={
        'console_scripts': [
            'aurora = aurora_cli.main:main',
        ]
    }
)