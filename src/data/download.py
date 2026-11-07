"""Unduhan ARFF OpenML melalui domain www.openml.org, tanpa mirror data lain."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

SOURCES = {41214: 'freMTPL2freq', 41215: 'freMTPL2sev'}
FREQ_COLUMNS = ['IDpol', 'ClaimNb', 'Exposure', 'Area', 'VehPower', 'VehAge',
                'DrivAge', 'BonusMalus', 'VehBrand', 'VehGas', 'Density', 'Region']


def read_arff(path):
    # OpenML uses a string attribute for VehGas, unsupported by scipy.io.arff.
    # These source files use dense CSV records with single-quoted strings.
    names = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.lower().startswith('@attribute '):
                names.append(stripped.split()[1].strip("'\""))
            if stripped.lower() == '@data':
                return pd.read_csv(handle, names=names, quotechar="'", na_values=['?'])
    raise ValueError('Header @data ARFF tidak ditemukan.')


def validate_schema(frame, name):
    expected = FREQ_COLUMNS if name == 'freMTPL2freq' else ['IDpol', 'ClaimAmount']
    if set(expected) != set(frame.columns) or frame.empty:
        raise ValueError(f'Skema {name} tidak sesuai.')
    if frame.IDpol.isna().any() or (frame.IDpol % 1 != 0).any():
        raise ValueError('ID polis harus bilangan bulat dan tidak hilang.')
    frame['IDpol'] = frame.IDpol.astype('int64')
    return frame


def load():
    return tuple(validate_schema(read_arff(f'data/raw/{name}.arff'), name) for name in SOURCES.values())


def download():
    folder = Path('data/raw')
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for identifier, name in SOURCES.items():
        metadata_url = f'https://www.openml.org/api/v1/json/data/{identifier}'
        response = requests.get(metadata_url, timeout=90)
        response.raise_for_status()
        metadata = response.json()['data_set_description']
        url = metadata['url'].replace('https://openml.org/', 'https://www.openml.org/')
        if not url.startswith('https://www.openml.org/'):
            raise ValueError('Sumber unduhan harus www.openml.org.')
        path = folder/f'{name}.arff'
        if not path.exists():
            response = requests.get(url, timeout=180)
            response.raise_for_status()
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(response.content)
            temporary.replace(path)
        payload = path.read_bytes()
        actual_md5 = hashlib.md5(payload).hexdigest()
        if actual_md5 != metadata['md5_checksum']:
            raise ValueError(f'Checksum {name} berbeda dari metadata OpenML.')
        frame = validate_schema(read_arff(path), name)
        records.append({'id': identifier, 'name': name, 'metadata_url': metadata_url,
                        'download_url': url, 'sha256': hashlib.sha256(payload).hexdigest(),
                        'md5': actual_md5, 'rows': len(frame), 'license': metadata['licence']})
        print(f'{name}: {len(frame)} baris, checksum dan skema valid.')
    freq, sev = load()
    # Sampel asli untuk inspeksi, bukan hasil analisis penuh.
    ids = freq.head(200).IDpol
    Path('data/sample').mkdir(parents=True, exist_ok=True)
    freq.loc[freq.IDpol.isin(ids)].to_csv('data/sample/frequency.csv', index=False)
    sev.loc[sev.IDpol.isin(ids)].to_csv('data/sample/severity.csv', index=False)
    Path('reports').mkdir(exist_ok=True)
    Path('reports/source.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    download()
