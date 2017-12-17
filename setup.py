from setuptools import find_packages, setup

print(find_packages())

setup(
    name='vecmaths',
    version="0.1",
    description="A collection of maths functions in Python.",
    author='Adam J Plowman, Maria S Yankova',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
)
