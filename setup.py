import os
from setuptools import setup, find_packages

packages = [x for x in find_packages('.') if x.startswith('coorx')]

setup(
    name = "coorx",
    version = "0.1.0",
    author = "Luke Campagnola",
    author_email = "lukec@alleninstitute.org",
    description = ("Coordinate system transforms"),
    license = "BSD",
    url = "http://github.com/campagnola/transforms",
    packages=packages,
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
)


