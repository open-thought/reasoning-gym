"""Setup script for rgc CLI tool."""

from setuptools import setup

setup(
    name="rgc",
    version="0.1.0",
    py_modules=["main"],
    install_requires=[
        "typer>=0.9.0",
        "rich>=13.7.0",
        "pyyaml>=6.0.1",
    ],
    entry_points={
        "console_scripts": [
            "rgc=main:main",
        ],
    },
)
