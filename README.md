# Agente Ollama

Projeto que integra o cliente Ollama com utilitários de RAG (retrieval-augmented
generation), gerenciamento de modelos e um wrapper de chat para criar agentes
conversacionais. Esta documentação apresenta instruções básicas de uso,
variáveis de ambiente esperadas e exemplos mínimos.

## Instalação

1. Crie e ative um ambiente virtual (recomendado):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Observe que `setup.py` lê `README.md` com encoding `utf-16` por compatibilidade
	com este repositório. Se ocorrerem erros, converta o `README.md` para UTF-8.

## Variáveis de ambiente

Coloque variáveis sensíveis em um arquivo `.env` (o projeto já pode ler esse
arquivo via `dotenv`). Exemplo de variáveis esperadas:

- `COMPLETION_MODEL` — modelo padrão para geração (ex: `qwen3:1.7b`)
- `EMBEDDING_MODEL` — modelo de embeddings (ex: `nomic-embed-text`)
- `OLLAMA_HOST` — URL do servidor Ollama (ex: `http://localhost:11434`)
- `BOTCITY_LOGIN`, `BOTCITY_KEY` — variáveis de exemplo usadas no projeto

Crie um `.env` local com as credenciais necessárias antes de executar.

## Uso rápido (exemplos mínimos)

Importação básica dos componentes principais:

```python
from agente_ollama.agente import Agente
from agente_ollama.rag.rag import RagIndex
from agente_ollama.chat.chat import Chat

# Inicializar agente com modelos padrão
bot = Agente()
print(bot.models)

# Criar/usar um índice RAG (exemplo simplificado)
# idx = RagIndex()
# idx.build(data_folder='path/to/docs')

# Criar chat (modo simplificado)
# chat = Chat(model='meu-modelo')
# resp = chat.chat('Olá, explique o RAG em 2 frases')
# print(resp)
```

## Notas e dependências

Dependências principais (ver `requirements.txt`): `ollama`, `PyPDF2`, `python-docx`,
`pandas`, `Pillow` e bibliotecas para processamento de PDFs/ imagens. Algumas
funcionalidades requerem um servidor Ollama rodando localmente.

## Contribuição

Documente o que alterar e adicione testes ou exemplos quando possível. Este
projeto é distribuído sob licença MIT (arquivo `LICENSE`).

---
Arquivo de documentação gerado automaticamente pelo assistente de documentação.
