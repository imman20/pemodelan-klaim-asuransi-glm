"""Laporan aktuaria dan README dari hasil eksekusi yang sama."""
import json


def write_reports(root, r, evaluation, severity, lift, double, sensitivity):
    audit = r['audit']
    method = f'''Saya mengaudit kedua tabel sumber penuh, lalu mengambil simple random sample
{r['sample_policies']:,} ID polis tanpa pengembalian dengan seed 2024 sebelum split 80:20.
Saya membatasi pelatihan untuk waktu dan memori lingkungan lokal; hasil ini bukan analisis
model penuh atas {audit['raw_policies']:,} polis. Sampel menghasilkan {r['train_policies']:,}
polis latih dan {r['test_policies']:,} polis uji, serta {r['train_claim_rows']:,} dan
{r['test_claim_rows']:,} baris klaim positif yang terhubung. Overlap ID = {r['overlap_policy_ids']}.
Semua klaim dari suatu ID mengikuti split polis tersebut. Saya tidak menstratifikasi
berdasarkan target dan tidak mengubah sampel berdasarkan hasil uji.

Saya mengagregasikan klaim positif per ID sebelum left join one-to-one ke frekuensi.
Saya menolak ID frekuensi berulang, target count noninteger/negatif, dan angka tidak hingga;
saya mengecualikan exposure <= 0 serta klaim <= 0 dengan jumlah tercatat dalam audit.
Exposure >1 tetap dipertahankan sebagai periode lebih dari satu tahun.
Baris klaim identik dipertahankan karena tidak tersedia claim ID untuk membuktikan duplikasi.
Tidak ada severity terhubung diperlakukan sebagai biaya tercatat nol, bukan bukti tidak ada
kewajiban klaim. Ketidakcocokan count tidak diperbaiki dengan menciptakan klaim.

Saya memakai Area, VehGas, Region, VehBrand sebagai faktor kategorik, serta VehPower,
VehAge, DrivAge, BonusMalus dan log(1+Density) sebagai efek numerik linier.
Saya menetapkan bentuk ini sebelum fit; ini pendekatan sederhana dan dapat melewatkan
nonlinieritas usia. Imputasi median/modus, standardisasi, dan kategori referensi dipelajari
dari polis latih saja. Kategori baru mendapat semua dummy nol dan peringatan pustaka;
prediksi demikian memakai efek referensi dan perlu ditinjau sebelum pemakaian nyata.
Kategori referensi: {json.dumps(r['references'], ensure_ascii=False)}.

Saya memodelkan jumlah klaim Poisson dengan log-link dan offset log(exposure).
Offset nol menghasilkan frekuensi tahunan. Saya membandingkan NB2, Var(N)=mu+alpha*mu²,
dengan alpha = max(1e-6, sum((N-mu)²-mu)/sum(mu²)) dari residual Poisson latih.
Ini estimator method-of-moments plug-in, bukan joint maximum likelihood.
Pearson chi-square/df Poisson = {r['train_pearson_dispersion']:.6f}; alpha NB2 =
{r['nb2_alpha_train_moments']:.6f}. Heterogenitas dan spesifikasi rerata juga dapat
menyebabkan overdispersi. Interval koefisien NB2 bersyarat pada alpha, tanpa ketidakpastian
estimasi alpha. AIC NB2 bawaan memperlakukan alpha tetap, sehingga saya menambahkan
penalti dua sebagai sensitivitas satu parameter tambahan; keduanya bukan bukti optimalitas MLE.

Saya mengestimasi GLM Gamma log-link pada setiap baris klaim positif, bukan rata-rata polis.
Beberapa klaim satu polis memberi beberapa baris dengan kovariat polis sama dan bobot satu
per klaim. Saya tidak membobot severity dengan exposure. Dependensi antarklaim satu polis
dapat membuat interval Wald konvensional terlalu sempit; saya tidak menafsirkannya sebagai
inferensi konfirmatori. IRLS langsung tidak stabil untuk target tanpa batas pada data asli.
Saya meminimumkan objective koefisien Gamma log-link yang konveks, mean(eta + y_scaled*exp(-eta)),
melalui trust-exact setelah membagi target dengan rerata latih. Saya mengembalikan skala
intersep, kemudian menjalankan IRLS statsmodels dari optimum tersebut untuk inferensi.
Saya mewajibkan gradien maksimum <=1e-6 dan konvergensi IRLS; diagnostik lengkap tersimpan
di `gamma_optimization` pada hasil.json. Saya tidak mengganti target atau memakai data uji
untuk mengatasi kegagalan numerik tersebut.
Saya melatih dua target terpisah: nilai klaim dibatasi di kuantil 99%
latih ({r['train_cap_q99']:.2f} EUR) dan nilai klaim asli tanpa batas. Kuantil tidak memakai
klaim uji. Saya juga menyajikan sensitivitas biaya terhadap q95, q99, maksimum latih, dan
tanpa batas. Hanya q99 dan tanpa batas memiliki model Gamma tersendiri; q95 merupakan
sensitivitas target, bukan hasil model yang dilatih ulang.

Premi murni tahunan = frekuensi tahunan × severitas per klaim. Biaya periode = premi murni
× exposure. Target terbatas berbeda dari target tanpa batas dan bukan estimasi cadangan
atau tarif komersial. Pembanding HistGradientBoosting memakai Poisson loss pada ClaimNb/Exposure
dengan sample_weight=Exposure, preprocessing fit pada latih dan 100 iterasi tetap tanpa
early stopping. Saya memakai Gamma q99 yang sama untuk kedua prediksi biaya agar perbedaan
berasal dari frekuensi. Poisson GLM ditetapkan sebagai model utama sebelum evaluasi,
bukan dipilih berdasarkan data uji.
'''
    definitions = '''Saya mengevaluasi frekuensi dengan mean Poisson deviance pada count periode
di polis uji yang sama. Saya juga melaporkan deviance rate berbobot exposure, yang memiliki
normalisasi berbeda; ini tidak boleh dianggap angka identik dengan deviance count.
Deviance NB2 juga dihitung untuk ketiga prediksi dengan alpha latih yang sama agar
perbandingan tetap memakai satu aturan penilaian. Ini bukan estimasi ulang alpha pada uji.
Gamma deviance dihitung per klaim uji positif untuk target yang sesuai. AIC frekuensi
dibandingkan hanya pada respons count dan sampel latih yang sama. AIC Gamma terbatas dan
tanpa batas dilaporkan terpisah, tidak diperingkatkan karena target berbeda. AIC tidak
dibandingkan lintas frekuensi/severitas atau terhadap boosting.

Saya membentuk desil dari akumulasi exposure sesudah mengurutkan premi murni GLM dari
rendah ke tinggi. Setiap polis utuh ditempatkan menurut titik tengah exposure sehingga
batas tidak memecah polis dan exposure tiap desil mendekati 10%. Aktual dan prediksi per
desil adalah jumlah biaya dibagi jumlah exposure, bukan rerata rate polis tanpa bobot.
Lift aktual = rate biaya desil / rate biaya seluruh sampel uji. Double lift mengurutkan
rasio prediksi biaya boosting/GLM, lalu membuat desil exposure dengan cara yang sama dan
menampilkan kedua prediksi serta aktual. Grafik double lift membantu melihat area
ketidaksepakatan model, bukan bukti kausal atau pengganti validasi eksternal.

Pada tabel relativitas, exp(beta) faktor adalah pengali terhadap kategori referensi dengan
kovariat lain tetap. Efek numerik adalah pengali untuk kenaikan satu simpangan baku latih,
bukan satu tahun/satu unit mentah. `numeric_sd_train` dalam hasil.json menyimpan konversinya;
relativitas satu unit mentah = exp(beta/sd). Intersep adalah rate/severity pada kategori
referensi dan numerik terstandar nol, bukan pengali tarif. Interval adalah Wald 95% bersyarat
spesifikasi. Standardisasi dan referensi dipakai ulang pada uji tanpa fit ulang.
'''
    results = f'''{evaluation.to_markdown(index=False)}

{severity.to_markdown(index=False)}

Total exposure uji = {r['test_exposure']:.4f} tahun polis.
Biaya terbatas aktual = EUR {r['test_actual_limited_cost']:,.2f}; prediksi GLM =
EUR {r['test_predicted_limited_cost']:,.2f}; prediksi boosting =
EUR {r['test_boost_predicted_limited_cost']:,.2f}.
Rasio prediksi biaya GLM/aktual terbatas = {r['test_limited_cost_calibration_ratio']:.4f},
sehingga total biaya belum terkalibrasi baik. Salah satu keterbatasan penting adalah
target frekuensi ClaimNb tidak selalu cocok dengan jumlah severity yang tercatat.
Biaya tanpa batas aktual = EUR {r['test_actual_unlimited_cost']:,.2f}; prediksi GLM =
EUR {r['test_predicted_unlimited_cost']:,.2f}.
Pembatasan menghilangkan EUR {r['train_claim_cost_removed_by_cap']:,.2f} pada latih dan
EUR {r['test_claim_cost_removed_by_cap']:,.2f} pada uji. Perbedaan total target ini harus
dipertimbangkan ketika menilai kalibrasi. Semua GLM konvergen: {r['all_glm_converged']}.

AIC Poisson latih = {r['poisson_train_aic']:.3f}; AIC NB2 alpha tetap =
{r['nb_train_aic_fixed_alpha']:.3f}; sensitivitas penalti alpha =
{r['nb_train_aic_plus_alpha_parameter']:.3f}.
'''
    limitations = '''Saya memakai data historis motor third-party liability Prancis yang tidak otomatis
mewakili pasar Indonesia. Sumber frekuensi dan severitas memiliki ketidakcocokan ID/count;
produk frekuensi × severity mengasumsikan keterkaitan target dan proses klaim yang belum
sepenuhnya terverifikasi. Evaluasi biaya memakai pembayaran positif yang tercatat, tidak
mencakup klaim hilang, inflasi, perkembangan klaim, deductible, limit kontrak, biaya akuisisi,
margin, pajak, reasuransi, atau perubahan exposure mendatang. Saya tidak mengoreksi perbedaan
count secara spekulatif. Tail berat dan sedikit klaim pada desil tertentu membuat kurva
biaya tidak stabil. Korelasi fitur, efek numerik linier, dan ketergantungan klaim membatasi
interpretasi serta interval. Split acak tunggal belum menguji generalisasi antarwaktu.
Saya hanya melatih sampel polis yang didokumentasikan; audit skema dan kualitas menggunakan
data sumber penuh. Studi ini merupakan portofolio dan belum menjadi tarif siap pakai
untuk perusahaan asuransi.
'''
    report = '''# Pemodelan Frekuensi dan Severitas Klaim Kendaraan

Saya menghubungkan frekuensi, severitas, exposure, dan biaya klaim melalui model yang dapat diperiksa.
Seluruh angka berasal dari eksekusi pipeline dan tersimpan dalam `reports/hasil.json` serta CSV.

## Audit sumber penuh

| Pemeriksaan | Hasil |
|---|---:|
''' + '\n'.join(f'| {key} | {value} |' for key, value in audit.items()) + f'''

Saya melaporkan baris klaim dan ID unik yang tidak terhubung secara terpisah. Mismatch adalah
perbandingan ClaimNb dengan jumlah baris severity positif per polis sesudah agregasi.
Kamus: [docs/data_dictionary.md](../docs/data_dictionary.md); sumber dan hash di `source.json`.

## Metodologi

{method}

## Definisi evaluasi dan relativitas

{definitions}

## Hasil uji dan AIC latih

{results}

![Lift](../assets/lift.png)

## Aktual versus prediksi dan lift

{lift.to_markdown(index=False)}

## Double lift

{double.to_markdown(index=False)}

## Sensitivitas klaim ekstrem

{sensitivity.to_markdown(index=False)}

Tabel relativitas seluruh koefisien, kategori referensi, dan interval: `relativities.csv`.
EDA berbobot exposure per faktor: `eda_rates.csv`. Prediksi per polis: `test_predictions.csv`.

## Keterbatasan

{limitations}
'''
    (root/'reports/laporan_analisis.md').write_text(report, encoding='utf-8')
    readme = f'''# Pemodelan Frekuensi dan Severitas Klaim Asuransi Kendaraan

Saya memodelkan jumlah dan nilai klaim kendaraan dengan GLM untuk menghitung premi murni berbasis exposure.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![CI](https://github.com/imman20/pemodelan-klaim-asuransi-glm/actions/workflows/ci.yml/badge.svg)
![MIT](https://img.shields.io/badge/License-MIT-green)

![Lift dan double lift](assets/lift.png)

## Ringkasan untuk recruiter

Saya mengaudit {audit['raw_policies']:,} polis sumber dan melatih model pada sampel {r['sample_policies']:,} polis yang dapat direproduksi.
Saya membandingkan Poisson, Negative Binomial, Gamma, dan gradient boosting dengan memperhitungkan exposure serta klaim ekstrem.
Saya menyajikan evaluasi uji, relativitas faktor, notebook yang dieksekusi, dan laporan teknis dengan keterbatasan aktuaria.

## Latar belakang dan pertanyaan

Saya memeriksa bagaimana faktor risiko berkaitan dengan frekuensi dan severitas, apakah
Poisson mengalami overdispersi, dan bagaimana klaim ekstrem memengaruhi premi murni.
Saya membandingkan akurasi dan kalibrasi model yang mudah diinterpretasikan dengan boosting.

## Data

[freMTPL2freq, OpenML 41214](https://www.openml.org/d/41214) dan
[freMTPL2sev, OpenML 41215](https://www.openml.org/d/41215), lisensi CC0.
Sitasi: C. Dutang dan A. Charpentier (2018), CASdatasets: Insurance datasets.
Unduhan dan checksum divalidasi oleh `src/data/download.py`; metadata sumber ada di
`reports/source.json`. Data penuh tidak dikomit, sampel asli ada di `data/sample/`.
Tabel frekuensi adalah satu baris polis, tabel severitas dapat memiliki beberapa klaim
per polis. [Kamus data dan hubungan](docs/data_dictionary.md).

## Metodologi

```mermaid
flowchart LR
  A[OpenML dan audit penuh] --> B[Agregasi klaim dan join satu polis]
  B --> C[Sampel ID dan split 80:20]
  C --> D[Latih: preprocessing dan kuantil klaim]
  D --> E[Poisson offset dan NB2]
  D --> F[Gamma terbatas dan tanpa batas]
  D --> G[Boosting Poisson berbobot exposure]
  E --> H[Uji: deviance, kalibrasi, lift dan double lift]
  F --> H
  G --> H
```

{method}

## Hasil utama

{results}

Saya membandingkan deviance pada target dan split yang sama. Model dengan deviance count
terkecil dalam tabel adalah pembanding akurasi uji, bukan otomatis rekomendasi tarif.
Saya menilai kalibrasi total melalui rasio prediksi/aktual dan kurva per desil; grafik biaya
dapat berfluktuasi karena target berekor berat. GLM menyediakan relativitas langsung,
sementara boosting mengakomodasi nonlinieritas dengan interpretasi yang lebih terbatas.
[Laporan teknis](reports/laporan_analisis.md), [relativitas](reports/relativities.csv),
[lift](reports/lift.csv), [double lift](reports/double_lift.csv).

## Struktur folder

```text
src/          data/download.py, data/prepare.py, models.py, pipeline.py, reporting.py, notebook.py
tests/        validasi, join, split, exposure, premi, dan smoke pipeline
notebooks/    analisis.ipynb dengan output eksekusi
reports/      hasil.json, laporan, evaluasi, relativitas, prediksi dan desil
assets/       grafik dari data asli
data/sample/  sampel kecil sumber
```

## Cara menjalankan

Python 3.11, dijalankan dari root repo:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate; Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
python -m ruff check .
python -m pytest -q
```

Perintah tunggal mengunduh kedua dataset, menjalankan pipeline sampel 100.000 polis, dan
mengeksekusi notebook. Untuk mengubah ukuran secara eksplisit: `python -m src.pipeline --max-policies 677991`,
lalu `python -m src.notebook`. Model penuh memerlukan lebih banyak memori dan waktu dan
belum dijalankan dalam hasil ini. CI memakai fixture kecil tanpa unduhan/pelatihan penuh.

## Keterbatasan

{limitations}

## Lisensi

Kode berlisensi [MIT](LICENSE) atas nama Immantaka Juniarsa. Data OpenML berlisensi CC0.

## Kontak

Immantaka Juniarsa, S1 Statistika Universitas Airlangga (2024).
[GitHub imman20](https://github.com/imman20), immantakajuniarsa30@gmail.com.
'''
    (root/'README.md').write_text(readme, encoding='utf-8')
