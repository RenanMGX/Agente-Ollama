from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-16') as f:
    requirements = [line for line in f.read().splitlines() if not line.startswith('#') and line.strip()]

setup(
    name="agente-ollama",
    version="0.1.1",
    packages=find_packages(),
    install_requires=requirements,
    description="A package for integrating Ollama with LangChain.",
    long_description=open(path.join(here, 'README.md'), encoding='utf-16').read(),
    long_description_content_type='text/markdown',
    author="Renan Oliveira",
    author_email="renanmgx@hotmail.com",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)