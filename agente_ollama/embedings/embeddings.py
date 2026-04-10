"""
Módulo de embeddings: utilitários para gerar e buscar embeddings usando
modelos Ollama.

Exemplos de uso:

    bot = Embedding(model='nomic-embed-text')
    base = bot.generate_list_embeddings_base(['doc1','doc2'])
    results = bot.search('consulta')
"""

import ollama
import numpy as np

from typing import Any, Callable, Literal, Mapping, Sequence, List, Dict, Union, Optional, Tuple
from copy import deepcopy
from datetime import date, datetime

from ollama._types import (
  ChatResponse,
  Tool,
)


class Embedding:
    """Classe para gerar e consultar uma base de embeddings.

    A instância mantém um dicionário interno (`embeddings_base`) com os vetores
    gerados. A base pode ser construída a partir de uma lista de strings ou um
    dicionário {chave: texto}.
    """

    def __init__(self, *, model: str, host: Literal["http://localhost:11434"] | str | None = None, **kwargs) -> None:
        """Inicializa o cliente Ollama para requests de embedding.

        Args:
            model (str): nome do modelo de embedding a ser usado.
            host (str | None): host do servidor Ollama (opcional).
        """
        self.model = model
        self.__client = ollama.Client(host, **kwargs)
        self.__embeddings_base: Dict[str, Any] | None = None

    @property
    def embeddings_base(self) -> Dict[str, Any] | None:
        """Retorna a base de embeddings atual (ou None se não gerada)."""
        return self.__embeddings_base

    @embeddings_base.setter
    def embeddings_base(self, value: Dict[str, Any]) -> None:
        """Define a base de embeddings validando formato.

        O dicionário deve conter a chave `list_embeddings` e uma das chaves
        `list_str` (lista ordenada) ou `dict_str` (mapeamento).
        """
        if not isinstance(value, dict):
            raise ValueError("O dicionário de embeddings deve ser do tipo dict.")

        if "list_embeddings" not in value:
            raise ValueError("O dicionário de embeddings deve conter a chave 'list_embeddings'.")

        if not any(key in value for key in ["list_str", "dict_str"]):
            raise ValueError("O dicionário de embeddings deve conter a chave 'list_str' ou 'dict_str'.")

        self.__embeddings_base = value

    def generate_list_embeddings_base(self, data_base: List[str] | Dict[str, Any]) -> Dict[str, List[float] | str]:
        """Gera a base de embeddings a partir de uma lista ou dicionário.

        Args:
            data_base (List[str] | Dict[str, str]): fonte de textos.

        Returns:
            dict: dicionário contendo `list_embeddings` e `list_str` ou `dict_str`.
        """
        embeddings_base: Dict[str, Any] = {}
        if isinstance(data_base, dict):
            embeddings_base["dict_str"] = data_base
            embeddings_base["list_embeddings"] = [ollama.embeddings(model=self.model, prompt=str(name)).embedding for name in data_base]

        elif isinstance(data_base, list):
            embeddings_base["list_str"] = data_base
            embeddings_base["list_embeddings"] = [ollama.embeddings(model=self.model, prompt=str(name)).embedding for name in data_base]

        self.__embeddings_base = embeddings_base
        return embeddings_base

    def search(self, query: str, *, list_str_base: List[str] | Dict[str, List[float]] | None = None, accuracy: float = 0) -> tuple:
        """Busca as strings mais semelhantes à `query` na base gerada.

        Args:
            query (str): texto de consulta.
            list_str_base (optional): se fornecido, gera a base a partir dele.
            accuracy (float): limiar mínimo de similaridade (0..1) para incluir.

        Returns:
            tuple: tupla de pares (string, similaridade) ordenada por similaridade.
        """
        if list_str_base is not None:
            self.generate_list_embeddings_base(list_str_base)

        if self.embeddings_base is None:
            raise ValueError("A base de embeddings não foi gerada. Por favor, forneça uma lista de strings ou um dicionário com as chaves 'list_str' e 'list_embeddings'.")

        q_emb = ollama.embeddings(model=self.model, prompt=str(query)).embedding

        sims = [np.dot(q_emb, d) / (np.linalg.norm(q_emb) * np.linalg.norm(d)) for d in self.embeddings_base["list_embeddings"]]

        ranked = sorted(
            zip((self.embeddings_base["list_str"] if "list_str" in self.embeddings_base else list(self.embeddings_base["dict_str"].keys())), sims),
            key=lambda x: x[1],
            reverse=True,
        )
        ranked = tuple([value if "list_str" in self.embeddings_base else (self.embeddings_base["dict_str"][value[0]], value[1]) for value in ranked if value[1] >= accuracy])

        return ranked


if __name__ == "__main__":
    bot = Embedding(
        model="snowflake-arctic-embed2:568m",
        host="http://patrimar-rpa2:11434/",
    )

    docs = [
        '2300 Rio de Janeiro',
        'Acqua Galleria Condomínio Resort - Condomínio 1',
        'Acqua Galleria Condomínio Resort - Condomínio 2',
        'Acqua Galleria Condomínio Resort - Condomínio 3',
    ]
    docs = {x: x for x in docs}
    bot.generate_list_embeddings_base(docs)

    while True:
        user_input = input("You: ")
        if user_input.lower() in [r"\exit", r"\quit", r"\e"]:
            break
        if not user_input:
            continue
        results = bot.search(user_input)
        print("Results:")
        try:
            res = results[0]
            if res[1] < 0.2:
                print("    No good match found.")
            else:
                for doc, score in results[0:3]:
                    print(f"    {doc} (similarity: {score:.4f})")
        except Exception:
            import traceback

            print(traceback.format_exc())
            import pdb

            pdb.set_trace()
    
    
    