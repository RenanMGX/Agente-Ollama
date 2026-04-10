"""
Módulo `agente` — wrapper simples para agrupar os modelos configurados.

Esta classe é uma conveniência para manter uma instância de
`agente_ollama.model_manager.models.Models` e expô-la para outros módulos.
Não altera comportamento de runtime — apenas encapsula a configuração de
modelos em um objeto `Agente`.
"""

from .model_manager.models import Models


class Agente:
    """Classe de alto nível que guarda uma instância de `Models`.

    Exemplos:
        >>> from agente_ollama.agente import Agente
        >>> bot = Agente()
        >>> print(bot.models)

    A propriedade `models` garante que somente instâncias de `Models` sejam
    atribuídas, lançando `ValueError` em caso contrário.
    """

    @property
    def models(self) -> Models:
        """Retorna a instância atual de `Models`.

        Returns:
            Models: objeto com as configurações de modelos (completion, embedding, etc.).
        """
        return self.__models

    @models.setter
    def models(self, value: Models) -> None:
        """Define a instância de `Models`.

        Args:
            value (Models): nova instância de `Models`.

        Raises:
            ValueError: se `value` não for uma instância de `Models`.
        """
        if isinstance(value, Models):
            self.__models = value
        else:
            raise ValueError("models must be an instance of Models class.")

    def __init__(self, model: Models = Models()):
        """Construtor.

        Args:
            model (Models, optional): instância de `Models`. Se não fornecida,
                uma nova instância será criada com `Models()`.
        """
        self.__models: Models = model


if __name__ == "__main__":
    bot = Agente()
    print(bot.models)
