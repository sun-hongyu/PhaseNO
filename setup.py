
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as build_ext_original
from subprocess import call


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="PhaseNO",
    version="0.1.0",
    author="",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    packages=["PhaseNo"],
    install_requires=[
        "lightning",
        "torch",
        "torch_geometric",
        #following are for example, not strictly necessary
        "ipykernel",
        "matplotlib",
        "numpy",
        "obspy",
        "pandas",
        "tqdm"
        ],
    python_requires=">=3.8",
    zip_safe=False,
    include_package_data=True,
)