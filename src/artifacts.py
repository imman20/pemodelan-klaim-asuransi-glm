"""Pertahankan hash artefak teks yang sama di Windows dan Linux."""


def normalize_text(paths):
    for path in paths:
        if path.is_file() and path.suffix in {'.csv', '.md', '.json', '.ipynb'}:
            path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
