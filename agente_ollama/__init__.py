"""Pacote principal `agente_ollama`.

Este __init__ adiciona o diretório do pacote ao sys.path para facilitar
imports quando o pacote é usado como um conjunto de módulos locais.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))