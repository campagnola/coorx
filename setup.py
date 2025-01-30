import os
import re

from setuptools import setup, find_packages

packages = [x for x in find_packages(".") if x.startswith("coorx")]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def get_version():
    init_py = os.path.join(os.path.dirname(__file__), "coorx", "__init__.py")
    with open(init_py, "r") as f:
        content = f.read()
    if version_match := re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M):
        return version_match[1]
    raise RuntimeError("Unable to find version string")


setup(
    name="coorx",
    version=get_version(),
    author="Luke Campagnola",
    author_email="lukec@alleninstitute.org",
    description="Object-oriented linear and nonlinear coordinate system transforms, plus coordinate system graphs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD",
    url="http://github.com/campagnola/coorx",
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
