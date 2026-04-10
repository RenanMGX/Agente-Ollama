"""
Instalação do pacote "agente-ollama".

Este arquivo `setup.py` é o empacotador do projeto e é usado apenas para
distribuição/instalação via setuptools. Ele lê as dependências e o
long_description a partir de `README.md`.

Observação importante:
- O projeto atual lê `README.md` com encoding 'utf-16' por compatibilidade
  histórica com este repositório. Caso encontre erros ao executar o
  setup, verifique o encoding do arquivo `README.md` (recomenda-se UTF-8).

Nota: este arquivo NÃO deve ser modificado para alterar comportamento do
pacote em runtime — apenas descreve metadados para instalação.
"""

from os import path
from setuptools import setup, find_packages

# Caminho absoluto para o diretório do projeto
here = path.abspath(path.dirname(__file__))

# Leitura simples das dependências a partir do README. O README pode
# conter linhas comentadas com '#' que serão ignoradas aqui.
with open(path.join(here, 'README.md'), encoding='utf-16') as f:
    requirements = [
        line for line in f.read().splitlines()
        if not line.startswith('#') and line.strip()
    ]

setup(
    name="agente-ollama",
    version="0.2.1",
    packages=find_packages(),
    install_requires=requirements,
    description="A package for integrating Ollama with LangChain.",
    long_description=open(path.join(here, 'README.md'), encoding='utf-16').read(),
    long_description_content_type='text/markdown',
    author="Renan Oliveira",
    author_email="renanmgx@hotmail.com",
    url="https://github.com/RenanMGX/Agente-Ollama",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)