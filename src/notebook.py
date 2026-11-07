"""Bangun dan eksekusi notebook audit dari artefak pipeline yang sama."""
import hashlib
import json
import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from src.artifacts import normalize_text


def main():
    root = Path.cwd()
    normalize_text([*Path('reports').glob('*'), Path('README.md')])
    manifest = {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                for folder in ['reports', 'assets'] for p in sorted((root/folder).glob('*')) if p.is_file()}
    Path('reports/artifact_manifest.json').unlink(missing_ok=True)
    manifest.pop('reports/artifact_manifest.json', None)
    Path('reports/artifact_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    cells = []
    cells.append(nbf.v4.new_markdown_cell(TITLE_TEXT))
    cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np
from IPython.display import display, Image
root = Path.cwd()
manifest = json.loads(Path('reports/artifact_manifest.json').read_text())
for file, expected in manifest.items():
    assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == expected, file
result = json.loads(Path('reports/hasil.json').read_text())
display(result)
"""))
    for title, description, table in SECTIONS:
        cells.append(nbf.v4.new_markdown_cell('## ' + title + '\n\n' + description))
        cells.append(nbf.v4.new_code_cell(f"table = pd.read_csv('reports/{table}')\ndisplay(table)"))
    cells.append(nbf.v4.new_markdown_cell('## Pemeriksaan konsistensi\n\nSaya menghitung ulang hubungan pokok dari prediksi tersimpan.'))
    cells.append(nbf.v4.new_code_cell(VERIFY_CODE))
    for figure in FIGURES:
        cells.append(nbf.v4.new_code_cell(f"display(Image(filename='assets/{figure}'))"))
    cells.append(nbf.v4.new_markdown_cell('## Interpretasi dan batas penggunaan\n\nSaya membaca hasil sebagai studi portofolio. Penjelasan lengkap, audit data, keputusan berbasis latih, dan keterbatasan tersedia di reports/laporan_analisis.md.'))
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    NotebookClient(notebook, timeout=300, resources={'metadata': {'path': str(root)}}).execute(env=env)
    Path('notebooks').mkdir(exist_ok=True)
    nbf.write(notebook, 'notebooks/analisis.ipynb')
    normalize_text([Path('reports/artifact_manifest.json'), Path('notebooks/analisis.ipynb')])
    print('Notebook dieksekusi dan seluruh hash artefak cocok.')



TITLE_TEXT = '# Frekuensi dan Severitas Klaim Kendaraan\n\nSaya meninjau artefak hasil eksekusi pipeline. Notebook ini memverifikasi hash sumber tabel/grafik dan menghitung ulang metrik penting tanpa melatih ulang pada data uji.'
SECTIONS = [('Frekuensi dengan exposure', 'Saya menghitung total klaim dibagi total exposure dari data latih, bukan rerata rate tanpa bobot.', 'eda_rates.csv'), ('Validasi count dan rate', 'Saya membandingkan model pada polis, respons count, dan exposure uji yang sama.', 'evaluation.csv'), ('Validasi severitas', 'Saya membedakan target terbatas dan tanpa batas. AIC lintas target tidak dibandingkan.', 'severity_evaluation.csv'), ('Relativitas dan interval', 'Saya membaca faktor terhadap kategori referensi dan numerik per satu simpangan baku latih.', 'relativities.csv'), ('Lift berbobot exposure', 'Saya mengurutkan premi GLM dan membentuk desil dari akumulasi exposure.', 'lift.csv'), ('Double lift', 'Saya mengurutkan rasio biaya boosting/GLM untuk melihat area ketidaksepakatan model.', 'double_lift.csv'), ('Sensitivitas tail', 'Saya memakai batas kuantil yang hanya dipelajari dari klaim latih; q95 bukan model refit.', 'tail_sensitivity.csv')]
FIGURES = ['frequency_area.png', 'lift.png']
VERIFY_CODE = 'from sklearn.metrics import mean_poisson_deviance\npred = pd.read_csv(\'reports/test_predictions.csv\')\nassert np.allclose(pred.predicted_count, pred.annual_frequency*pred.Exposure)\nassert np.allclose(pred.period_cost, pred.annual_pure_premium*pred.Exposure)\nassert np.isclose(pred.limited_amount.sum(), result[\'test_actual_limited_cost\'])\nassert np.isclose(mean_poisson_deviance(pred.ClaimNb, pred.predicted_count), result[\'frequency_evaluation\'][0][\'test_count_poisson_deviance\'])\nsplit = pd.read_csv(\'reports/split.csv\')\nassert not set(split.query("split == \'train\'").IDpol) & set(split.query("split == \'test\'").IDpol)\nprint(\'Count, rate, exposure, biaya, deviance, dan pemisahan ID konsisten.\')'

if __name__ == '__main__':
    main()
