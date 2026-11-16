# Reproduksi Eksperimen Churn Berbasis Graf Bipartit

Repositori ini memuat kode yang digunakan untuk eksperimen artikel tentang prediksi churn telekomunikasi menggunakan representasi graf bipartit pelanggan–nilai-fitur. Implementasi pembangkitan *random walk* dan pelatihan *skip-gram with negative sampling* ditulis secara eksplisit menggunakan NumPy dalam `Eksperimen_Komentar5.py`.

Repositori publik: <https://github.com/zakifiam20-crypto/customer-feature-bipartite-churn>

## Dataset

Penelitian menggunakan berkas `Telco_customer_churn.xlsx`, yaitu versi 33 kolom dari *Telco Customer Churn: IBM Dataset*. Dataset tidak didistribusikan ulang dalam repositori ini. Unduh dataset dari:

- IBM Cognos Analytics 11.1.3 sample: <https://community.ibm.com/community/user/blogs/steven-macko/2019/07/10/new-base-samples-for-ibm-cognos-analytics-1113>
- Kaggle mirror (Yean Zc): <https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset>

Letakkan `Telco_customer_churn.xlsx` di direktori yang sama dengan `Eksperimen_Komentar5.py` sebelum menjalankan eksperimen.

## Isi paket

- `Eksperimen_Komentar5.py`: implementasi utama eksperimen, termasuk pembentukan matriks indikator 55 kolom, truncated SVD, Node2Vec, MetaPath2Vec, fusi representasi, optimasi threshold dalam inner-CV, dan analisis sensitivitas.
- `Eksperimen_Komentar5.ipynb`: versi notebook dari eksperimen utama.
- `final_node2vec_metapath2vec_oof_predictions.csv`: indeks dan label training yang dipakai untuk menjaga pembagian data tetap sama dengan eksperimen utama.
- `outputs/`: hasil ringkas eksperimen yang dilaporkan dalam artikel.

## Lingkungan komputasi

Disarankan menggunakan Python 3.11 atau versi yang kompatibel. Pasang dependensi dengan:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Menjalankan eksperimen

```bash
python Eksperimen_Komentar5.py
```

Eksperimen menggunakan lima seed embedding (`42`–`46`). Konfigurasi utama memakai 10 *walk* per simpul pelanggan, panjang *walk* 80, dimensi embedding 32, dan batas pasangan konteks yang sama untuk Node2Vec dan MetaPath2Vec. Waktu komputasi dapat berlangsung lebih dari satu jam, bergantung pada perangkat keras.

## Peran Neo4j

Neo4j tidak digunakan untuk membentuk *random walk*, mempelajari embedding, melatih klasifikator, atau menghitung hasil evaluasi. Neo4j hanya digunakan secara terpisah untuk inspeksi visual struktur graf dan karena itu tidak diperlukan untuk mereproduksi hasil artikel.

