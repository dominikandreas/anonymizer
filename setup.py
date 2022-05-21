#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name='uai-anonymizer',
    version='2022.05.17',
    packages=find_packages(exclude=['test', 'test.*', 'images']),

    install_requires=[
        'pytest==3.9.1',
        'flake8==3.5.0',
        'Pillow==5.3.0',
        'requests==2.20.0',
        'googledrivedownloader==0.3',
        'progiter~=0.1.4'
    ],

    dependency_links=[],
)
