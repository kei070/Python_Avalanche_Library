# setup.py
from setuptools import setup, find_packages
from os import path

wd = path.abspath(path.dirname(__file__))

with open(path.join(wd, "README.md"), encoding="utf-8") as f:
    l_desc = f.read()
# end with

setup(
    name='ava_functions',
    version='0.0.1',
    author="Kai-Uwe Eiselt",
    license="GNU",
    description="Functions to help the avalanche danger level machine learning application.",
    long_description=l_desc,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[],
)