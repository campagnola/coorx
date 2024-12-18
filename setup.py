import os
from setuptools import setup, find_packages

packages = [x for x in find_packages('.') if x.startswith('coorx')]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name = "coorx",
    version = "1.0.0",
    author = "Luke Campagnola",
    author_email = "lukec@alleninstitute.org",
    description = ("Object-oriented coordinate system transforms in pure Python"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license = "BSD",
    url = "http://github.com/campagnola/coorx",
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
