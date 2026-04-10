from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import ollama


class RagIndex:
    """Índice RAG (Retrieval-Augmented Generation) baseado em embeddings Ollama.

    Fluxo típico:
        # Construir e salvar
        idx = RagIndex.build(
            files_paths=["doc.pdf", "manual.docx"],
            save_path="meu_rag.json",
            embedding_model="nomic-embed-text",
        )
        # Ou carregar de disco
        idx = RagIndex.load("meu_rag.json")
        # Buscar chunks relevantes
        chunks = idx.search("qual é o prazo de entrega?", top_k=5)
    """

    def __init__(
        self,
        chunks: List[str],
        embeddings: np.ndarray,
        model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._chunks: List[str] = chunks
        self._embeddings: np.ndarray = embeddings  # (N, D) float32
        self._model: str = model
        self._chunk_size: int = chunk_size
        self._chunk_overlap: int = chunk_overlap

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _split_by_separator(texts: List[str], sep: str, chunk_size: int) -> List[str]:
        """Divide cada texto pelo separador, mantendo partes menores que chunk_size intactas."""
        result: List[str] = []
        for text in texts:
            if len(text) <= chunk_size:
                result.append(text)
                continue
            parts = text.split(sep)
            current = ""
            for part in parts:
                candidate = (current + sep + part).lstrip(sep) if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        result.append(current)
                    current = part
            if current:
                result.append(current)
        return result

    @staticmethod
    def _chunk_text(
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> List[str]:
        """Recursive Text Splitter sem dependências externas.

        Hierarquia de separadores: \\n\\n → \\n → '. ' → caracteres
        Aplica overlap entre chunks consecutivos.
        """
        if not text.strip():
            return []

        # Etapa 1-3: divisão recursiva pelos separadores
        chunks: List[str] = [text]
        for sep in ["\n\n", "\n", ". "]:
            if all(len(c) <= chunk_size for c in chunks):
                break
            chunks = RagIndex._split_by_separator(chunks, sep, chunk_size)

        # Etapa 4: divisão forçada para chunks ainda grandes
        final: List[str] = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                final.append(chunk)
            else:
                start = 0
                while start < len(chunk):
                    final.append(chunk[start : start + chunk_size])
                    start += chunk_size
        chunks = final

        # Aplicar overlap
        if chunk_overlap <= 0 or len(chunks) <= 1:
            return [c.strip() for c in chunks if c.strip()]

        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-chunk_overlap:]
            merged = (prev_tail + chunks[i]).strip()
            overlapped.append(merged)

        return [c for c in overlapped if c.strip()]

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @staticmethod
    def build(
        files_paths: List[str],
        save_path: str,
        embedding_model: str,
        host: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        format: Literal["json", "npz"] = "json",
    ) -> "RagIndex":
        """Constrói o índice RAG a partir de uma lista de arquivos, salva e retorna.

        Args:
            files_paths: Caminhos dos arquivos de origem (txt, md, pdf, docx, xlsx, etc.).
            save_path: Caminho do arquivo de saída (ex: ``"meu_rag.json"`` ou ``"meu_rag.npz"``).
                       A extensão é adicionada automaticamente se não corresponder ao formato.
            embedding_model: Nome do modelo de embedding Ollama.
            host: Host do servidor Ollama (None = padrão local).
            chunk_size: Tamanho máximo de cada chunk em caracteres.
            chunk_overlap: Sobreposição entre chunks consecutivos em caracteres.
            format: Formato do arquivo de saída — ``'json'`` ou ``'npz'``.

        Returns:
            :class:`RagIndex` carregado em memória.
        """
        # Import local para evitar dependência circular (chat.py também importa de cá)
        from agente_ollama.chat.chat import extrair_texto  # type: ignore

        client = ollama.Client(host) if host else ollama.Client()

        all_chunks: List[str] = []
        for fp in files_paths:
            try:
                _type, content, _imgs = extrair_texto(fp)
                if _type == "image":
                    # imagens puras não têm texto para indexar
                    continue
                chunks = RagIndex._chunk_text(content, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"[RAG build] falha ao processar '{fp}': {e}"
                )

        if not all_chunks:
            raise ValueError(
                "Nenhum chunk gerado. Verifique os arquivos fornecidos em 'files_paths'."
            )

        embeddings_list: List[Any] = []
        for chunk in all_chunks:
            emb = client.embeddings(model=embedding_model, prompt=chunk).embedding
            embeddings_list.append(list(emb))

        embeddings_np = np.array(embeddings_list, dtype=np.float32)

        # Garantir extensão correta
        save_p = Path(save_path)
        expected_ext = f".{format}"
        if save_p.suffix.lower() != expected_ext:
            save_p = save_p.with_suffix(expected_ext)

        save_p.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            payload: Dict[str, Any] = {
                "model": embedding_model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunks": all_chunks,
                "embeddings": embeddings_np.tolist(),
            }
            with open(save_p, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        elif format == "npz":
            np.savez_compressed(
                str(save_p),
                embeddings=embeddings_np,
                chunks=np.array(all_chunks, dtype=object),
                model=np.array(embedding_model),
                chunk_size=np.array(chunk_size),
                chunk_overlap=np.array(chunk_overlap),
            )
        else:
            raise ValueError(f"Formato não suportado: '{format}'. Use 'json' ou 'npz'.")

        return RagIndex(
            chunks=all_chunks,
            embeddings=embeddings_np,
            model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @staticmethod
    def load(path: str) -> "RagIndex":
        """Carrega um índice RAG de disco (auto-detecta JSON ou NPZ pela extensão).

        Args:
            path: Caminho do arquivo salvo por :meth:`build`.

        Returns:
            :class:`RagIndex` pronto para uso em :meth:`search`.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo RAG não encontrado: '{path}'")

        ext = p.suffix.lower()

        if ext == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return RagIndex(
                chunks=data["chunks"],
                embeddings=np.array(data["embeddings"], dtype=np.float32),
                model=data["model"],
                chunk_size=data["chunk_size"],
                chunk_overlap=data["chunk_overlap"],
            )

        if ext == ".npz":
            data = np.load(str(p), allow_pickle=True)
            return RagIndex(
                chunks=list(data["chunks"]),
                embeddings=data["embeddings"].astype(np.float32),
                model=str(data["model"]),
                chunk_size=int(data["chunk_size"]),
                chunk_overlap=int(data["chunk_overlap"]),
            )

        raise ValueError(
            f"Extensão '{ext}' não reconhecida. Use '.json' ou '.npz'."
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        embedding_model: Optional[str] = None,
        top_k: int = 5,
        host: Optional[str] = None,
    ) -> List[str]:
        """Retorna os ``top_k`` chunks mais relevantes para a query.

        Args:
            query: Texto da consulta.
            embedding_model: Modelo de embedding para a query. Se None, usa o
                modelo registrado no índice (:attr:`_model`).
            top_k: Número máximo de chunks a retornar.
            host: Host do servidor Ollama (None = padrão local).

        Returns:
            Lista de strings com os chunks mais relevantes, ordenados por
            similaridade decrescente.
        """
        model = embedding_model or self._model
        client = ollama.Client(host) if host else ollama.Client()

        q_emb = np.array(
            client.embeddings(model=model, prompt=query).embedding, dtype=np.float32
        )

        # Similaridade cosseno vetorizada
        norms = np.linalg.norm(self._embeddings, axis=1)
        q_norm = np.linalg.norm(q_emb)
        # Evita divisão por zero
        valid = (norms > 0) & (q_norm > 0)
        sims = np.zeros(len(self._chunks), dtype=np.float32)
        if q_norm > 0:
            sims[valid] = (self._embeddings[valid] @ q_emb) / (norms[valid] * q_norm)

        top_indices = np.argsort(sims)[::-1][:top_k]
        return [self._chunks[i] for i in top_indices if sims[i] > 0]

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RagIndex(chunks={len(self._chunks)}, "
            f"model='{self._model}', "
            f"chunk_size={self._chunk_size}, "
            f"chunk_overlap={self._chunk_overlap})"
        )
