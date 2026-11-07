# Sumber dan hubungan data

Saya memakai freMTPL2freq (OpenML ID 41214) dan freMTPL2sev (ID 41215), lisensi CC0.
Sitasi: C. Dutang dan A. Charpentier (2018), CASdatasets: Insurance datasets.
Metadata, URL unduhan, MD5 yang diverifikasi, SHA256 dan ukuran baris tersimpan di
`reports/source.json`. Skrip menggunakan `https://www.openml.org/api/v1/json/data/41214`
dan `/41215`, lalu endpoint unduhan www.openml.org yang dinyatakan metadata.
Skrip tidak mengganti data apabila unduhan ditolak.

| Kolom | Unit dan arti |
|---|---|
| IDpol | ID polis integer, kunci unik di frekuensi; berulang di severitas |
| ClaimNb | Jumlah klaim selama periode exposure; integer nonnegatif |
| Exposure | Lama pertanggungan dalam tahun polis; offset log(exposure) |
| Area | Kode area A-F |
| VehPower | Kelas daya kendaraan |
| VehAge | Umur kendaraan dalam tahun |
| DrivAge | Umur pengemudi dalam tahun |
| BonusMalus | Indeks bonus-malus |
| VehBrand | Kode merek kendaraan |
| VehGas | Jenis bahan bakar, Diesel atau Regular |
| Density | Kepadatan penduduk wilayah, ditransformasi log(1+Density) |
| Region | Kode wilayah Prancis |
| ClaimAmount | Nilai satu klaim dalam EUR, unit analisis Gamma |

Frekuensi memiliki satu baris per polis. Severitas dapat memiliki banyak baris
per polis dan tidak memiliki claim ID terpisah. Saya menghitung `severity_count`
dan `total_amount` melalui groupby IDpol sebelum join one-to-one. Exposure tidak
dijumlah ulang dari hasil join baris klaim. Untuk fit Gamma saja, klaim positif
di-join many-to-one dengan kovariat polis; baris berganda di sini memang unit klaim.

Saya memisahkan ID sebelum mengambil baris klaim latih/uji. Klaim dengan ID yang
tidak ada di frekuensi dicatat dalam audit dan tidak memiliki kovariat untuk fit.
Perbedaan ClaimNb versus jumlah klaim positif dicatat, bukan dipaksa sama. Klaim
identik dipertahankan karena bisa merupakan kejadian berbeda. Nilai biaya yang
tidak terhubung setelah left join menjadi nol untuk target pembayaran tercatat;
ini bukan asumsi bahwa klaim yang hilang tidak pernah terjadi.

Target biaya terbatas menjumlah min(ClaimAmount, q99_latih) per klaim. Target
tanpa batas menjumlah nilai asli positif. Premi murni tahunan dikalikan exposure
untuk prediksi pembayaran selama observasi. Tidak tersedia informasi inflasi,
perkembangan klaim, deductible, limit kontrak, maupun tanggal penilaian aktuaria.

Sampel di `data/sample/` adalah 200 polis pertama sumber dan klaim terhubung,
hanya untuk inspeksi skema. Sampel analisis adalah simple random sample 100.000
ID yang berbeda dan dijelaskan di laporan. Fixture tes adalah data sintetis kecil
yang tidak dipakai menghasilkan angka portofolio atau mengganti dataset publik.
