import zipfile, tempfile, shutil, os, re, pathlib

def sanitize_docx(path, out_path=None):
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
