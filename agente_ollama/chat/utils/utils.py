"""
Utilitários auxiliares para processamento de documentos usados pelo pacote
`agente_ollama.chat`.

Atualmente contém:
- sanitize_docx: remove atributos problemáticos de arquivos `.docx`
  (por exemplo, atributos longos `o:gfxdata`) reempacotando o documento.
"""

import zipfile, tempfile, shutil, os, re, pathlib


def sanitize_docx(path, out_path=None):
    """Sanitiza um arquivo .docx removendo atributos binários muito longos.

    Muitos arquivos `.docx` podem conter atributos XML com dados binários
    (ex: `o:gfxdata`) que atrapalham bibliotecas de parsing. Esta função:

    1. Extrai o `.docx` (zip) para uma pasta temporária;
    2. Remove atributos com padrões `*gfxdata="..."` do XML `word/document.xml`;
    3. Reempacota o conteúdo em um novo `.docx` (`out_path`).

    Args:
        path (str | Path): caminho para o `.docx` original.
        out_path (str | Path, optional): caminho de saída. Por padrão, usa o
            mesmo nome com sufixo `_sanitized`.

    Returns:
        str | Path: caminho para o arquivo `.docx` sanitizado.

    Raises:
        FileNotFoundError: se o arquivo ou o `word/document.xml` não forem encontrados.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if out_path is None:
        out_path = p.with_name(p.stem + "_sanitized" + p.suffix)
    tmp = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(p, 'r') as z:
            z.extractall(tmp)
        doc_xml = os.path.join(tmp, 'word', 'document.xml')
        if not os.path.exists(doc_xml):
            raise FileNotFoundError("word/document.xml não encontrado no .docx")
        # leia e remova atributos o:gfxdata muito longos (não faz parsing XML)
        with open(doc_xml, 'rb') as f:
            data = f.read()
        text = data.decode('utf-8', errors='ignore')
        # remove atributos como:  o:gfxdata="...."
        new_text = re.sub(r'\s+o:gfxdata="[^"]+"', '', text)
        # opcional: se houver outros nomes com gfxdata (ajuste se necessário)
        new_text = re.sub(r'\s+[a-zA-Z0-9:_-]+gfxdata="[^"]+"', '', new_text)
        with open(doc_xml, 'wb') as f:
            f.write(new_text.encode('utf-8'))
        # reempacotar em novo .docx
        with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(tmp):
                for fname in files:
                    full = os.path.join(root, fname)
                    arc = os.path.relpath(full, tmp).replace(os.path.sep, '/')
                    z.write(full, arc)
        return out_path
    finally:
        shutil.rmtree(tmp)
