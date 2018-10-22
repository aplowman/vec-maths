"""Pip installation script."""

from setuptools import find_packages, setup

setup(
    name='vecmaths',
    version="0.1.0",
    description="A collection of (mostly vectorised) maths functions in Python.",
    author='Adam J Plowman, Maria S Yankova',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    classifiers=[
        'Development Status :: 4 - Beta'
    ],
)
