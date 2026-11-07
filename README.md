# Pemodelan Frekuensi dan Severitas Klaim Asuransi Kendaraan

Saya memodelkan jumlah dan nilai klaim kendaraan dengan GLM untuk menghitung premi murni berbasis exposure.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![CI](https://github.com/imman20/pemodelan-klaim-asuransi-glm/actions/workflows/ci.yml/badge.svg)
![MIT](https://img.shields.io/badge/License-MIT-green)

![Lift dan double lift](assets/lift.png)

## Ringkasan untuk recruiter

Saya mengaudit 678,013 polis sumber dan melatih model pada sampel 100,000 polis yang dapat direproduksi.
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

Saya mengaudit kedua tabel sumber penuh, lalu mengambil simple random sample
100,000 ID polis tanpa pengembalian dengan seed 2024 sebelum split 80:20.
Saya membatasi pelatihan untuk waktu dan memori lingkungan lokal; hasil ini bukan analisis
model penuh atas 678,013 polis. Sampel menghasilkan 80,000
polis latih dan 20,000 polis uji, serta 3,091 dan
800 baris klaim positif yang terhubung. Overlap ID = 0.
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
Kategori referensi: {"Area": "A", "VehGas": "Diesel", "Region": "R11", "VehBrand": "B1"}.

Saya memodelkan jumlah klaim Poisson dengan log-link dan offset log(exposure).
Offset nol menghasilkan frekuensi tahunan. Saya membandingkan NB2, Var(N)=mu+alpha*mu²,
dengan alpha = max(1e-6, sum((N-mu)²-mu)/sum(mu²)) dari residual Poisson latih.
Ini estimator method-of-moments plug-in, bukan joint maximum likelihood.
Pearson chi-square/df Poisson = 2.867204; alpha NB2 =
0.846232. Heterogenitas dan spesifikasi rerata juga dapat
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
latih (18056.02 EUR) dan nilai klaim asli tanpa batas. Kuantil tidak memakai
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


## Hasil utama

| model             |   test_count_poisson_deviance |   test_nb2_deviance_at_train_alpha |   test_exposure_weighted_rate_deviance |   actual_claims |   predicted_claims |   count_calibration_ratio |
|:------------------|------------------------------:|-----------------------------------:|---------------------------------------:|----------------:|-------------------:|--------------------------:|
| Poisson GLM       |                      0.334928 |                           0.296648 |                               0.633153 |            1114 |            1073.58 |                  0.963716 |
| NB2 GLM           |                      0.334934 |                           0.29663  |                               0.633165 |            1114 |            1085.41 |                  0.974332 |
| Gradient boosting |                      0.319576 |                           0.281785 |                               0.604132 |            1114 |            1077.31 |                  0.967064 |

| target    |   test_claim_rows |   gamma_deviance |   actual_mean |   predicted_mean |   train_aic |
|:----------|------------------:|-----------------:|--------------:|-----------------:|------------:|
| limited   |               800 |          1.05377 |       1533.49 |          1630.05 |     52987.7 |
| unlimited |               800 |          1.3971  |       1654.38 |          2342.04 |     62790   |

Total exposure uji = 10579.6686 tahun polis.
Biaya terbatas aktual = EUR 1,226,790.73; prediksi GLM =
EUR 1,740,362.67; prediksi boosting =
EUR 1,741,332.45.
Rasio prediksi biaya GLM/aktual terbatas = 1.4186,
sehingga total biaya belum terkalibrasi baik. Salah satu keterbatasan penting adalah
target frekuensi ClaimNb tidak selalu cocok dengan jumlah severity yang tercatat.
Biaya tanpa batas aktual = EUR 1,323,503.02; prediksi GLM =
EUR 2,484,115.56.
Pembatasan menghilangkan EUR 2,598,338.10 pada latih dan
EUR 96,712.29 pada uji. Perbedaan total target ini harus
dipertimbangkan ketika menilai kalibrasi. Semua GLM konvergen: True.

AIC Poisson latih = 33739.491; AIC NB2 alpha tetap =
33632.137; sensitivitas penalti alpha =
33634.137.


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
# Windows: .venv\Scripts\activate; Linux/macOS: source .venv/bin/activate
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

Saya memakai data historis motor third-party liability Prancis yang tidak otomatis
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


## Lisensi

Kode berlisensi [MIT](LICENSE) atas nama Immantaka Juniarsa. Data OpenML berlisensi CC0.

## Kontak

Immantaka Juniarsa, S1 Statistika Universitas Airlangga (2024).
[GitHub imman20](https://github.com/imman20), immantakajuniarsa30@gmail.com.
