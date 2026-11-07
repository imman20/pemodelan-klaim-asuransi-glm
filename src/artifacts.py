"""Pertahankan hash artefak teks yang sama di Windows dan Linux."""


def normalize_text(paths):
    for path in paths:
        if path.is_file() and path.suffix in {'.csv', '.md', '.json', '.ipynb'}:
            content = path.read_bytes().replace(b'\r\n', b'\n')
            if path.suffix == '.md':
                content = content.rstrip(b'\n') + b'\n'
            path.write_bytes(content)
