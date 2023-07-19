# Laporan Proyek Machine Learning - Hisyam Athaya

## Domain Proyek

Proyek ini membahas mengenai prediksi keterminuman (*potability*) atau potabilitas air. Ketersediaan air minum merupakan masalah lanjutan dari ketersediaan air. Tidak semua air bisa diminum, masyarakat yang hidup di wilayah industri memiliki permasalahan air yang tercemar dari limbah dan kotoran buangan pabrik. Beberapa penelitian menunjukkan bahwa sumber air yang tercemar adalah permasalahan besar yang dihadapi banyak negara, salah satunya India [1] dan China [2].

Salah satu implementasi teknologi dalam penanganan sumber daya air adalah monitoring dan manajemen air secara online. Pada projek ini, fokus adalah pengolahan data hasil dari sistem monitoring air. Pengolahan data sampel air dengan menggunakan model *machine learning* adalah salah satu solusi yang membantu mempercepat pemrosesan sampel air [3].  

## Business Understanding

### Problem Statements

Untuk mempersempit fokus pada penelitian ini, disusunlah pertanyaan penelitian sebagai berikut:

- Bagaimana memprediksi potabilitas air menggunakan Algoritma *Random Forest*?
- Bagaimana akurasi model tersebut dievaluasi?
- Apakah model prediksi air menggunakan Algoritma *Random Forest* sudah tepat?

### Goals

Tujuan dari projek ini:

- Mengembangkan model prediksi potabilitas Air menggunakan Algoritma *Random Forest*
- Perbandingan akurasi model dengan algoritma lain.
- Evaluasi model yang telah dibuat.

## Data Understanding

Dataset sampel air ini diperoleh dari penelitian oleh Gao, et al. mengenai potabilitas air di India [3], yang diperoleh dari beberapa titik sensor pada fasilitas pengolahan air. Dataset disimpan pada repository Kaggle dan pada projek ini diolah menggunakan Google Colaboratory. 

1. ph: tingkat pH pada air (0-14).
2. Hardness: Kapasitas air untuk mengendapkan sabun dalam miligram per liter (mg/L) .
3. Solids: Total dari bahan solid yang terlarutkan dalam part per million (ppm).
4. Chloramines: Jumlah chloramine dalam ppm.
5. Sulfate: Jumlah kandungan sulfat dalam mg/L.
6. Conductivity: Konduksivitas air dalam microsecond per centimeter (μS/cm).
7. Organic_carbon: Jumlah kandungan karbon organik dalam ppm.
8. Trihalomethanes: Jumlah Trihalomethanes dalam microgram per liter (μg/L).
9. Turbidity: Ukuran properti pemantul cahaya dalam satuan NTU.
10. Potability: Indeks keterminuman. 1 dapat diminum dan 0 tidak dapat diminum.  

## Data Preparation

Pada bagian ini dilakukan bersamaan dengan Analisis Data Eksploratori. Langkah-langkah yang dilakukan adalah:

1. Pemeriksaan dimensi dataset
Dari hasil pemeriksaan, terdapat 3276 sampel yang tersebar pada 10 fitur.

2. Pemetaan Tipe Data Fitur
Pemetaan Fitur pada dataset ini adalah:

Tabel 1. Fitur pada Dataset

| Fitur           | Tipe Data |
| --------------- | --------- |
| ph              | float64   |
| Hardness        | float64   |
| Solids          | float64   |
| Chloramines     | float64   |
| Sulfate         | float64   |
| Conductivity    | float64   |
| Organic_carbon  | float64   |
| Trihalomethanes | float64   |
| Turbidity       | float64   |
| Potability      | int64     |

Dapat disimpulkan dari Tabel 1 bahwa terdapat 10 fitur, dengan sembilan fitur bersifat *float* dan 1 fitur berisi integer, yaitu `Potability` yang berisi dua angka 0 dan 1. 0 bermakna air tidak aman diminum dan 1 bermakna air dapat diminum.

3. Pemeriksaan *Null Value*
Pemeriksaan *Null Value* dilakukan untuk memeriksa adanya nilai yang kosong pada dataset. Dari hasil pemeriksaan diperoleh hasil sebagai berikut. Pencarian digunakan dengan fungsi `is_null()`. 

* ph: 491 data kosong
* Sulfate: 781 data kosong
* Trihalomethanes: 162 data kosong

4. Visualisasi *Null Value*
Visualisasi ini dilakukan untuk mengetahui apakah ada pola tertentu, dari hasil yang diberikan tidak ada bentuk pola tertentu. Pada projek ini menggunakan visualisasi *heatmap* yang ditampilkan pada Gambar 1.

Gambar 1. *Null Value* pada data.

![Null Value](https://github.com/athayahisyam/dcdshit/assets/31662860/69b31c3d-cfaa-4cbe-aa4c-3de3791e7c9f)

Dapat dilihat dari gambar 1, bahwa *null value* muncul tidak dalam pola tertentu, namun tersebar secara acak pada data.

5. Pengisian Null Value dengan Imputasi
Untuk kasus ini, digunakan Imputasi berbasis Model dengan Model *Random Forest*. Digunakan demikian karena dataset ini akan menghasilkan nilai prediktif pada potabilitas (keterminuman) air, sehingga memiliki hipotesa yang kuat bahwa variabel dalam dataset ini saling berkorelasi satu sama lain.

7. Pemeriksaan *Skewness* pada Fitur Numerik dan Pemeriksaan *Outlier*

Hasil dari pemeriksaan *skewness* dan *box plot* ditunjukkan pada Gambar 2, 3, dan 4.

Gambar 2. Histogram Dataset

![Skewness](https://github.com/athayahisyam/dcdshit/assets/31662860/84411060-34d0-4857-b47a-856955121f32)

Pada tahap ini ditemukan bahwa fitur `Solids` memiliki skewness atau tendensi ke kanan, sehingga dilakukan transformasi *square root*. Alasan penggunaan adalah transformasi tersebut menjaga (*retain*) urutan besaran angka. Hasil setelah dilakukan proses transformasi *square root* ditunjukkan pada Gambar 3.

Gambar 3. Hasil *Square Root Transformation* pada Solid "meluruskan" *skewness* pada `Solid`

![skewness2](https://github.com/athayahisyam/dcdshit/assets/31662860/17b17eb6-c756-45ee-840a-55cebe142280)

Pada histogram di Gambar 3, dapat dilihat `Solids` sudah memiliki *skewness* yang seimbang. 

Gambar 4. *Box Plot* dari Dataset

![boxplot](https://github.com/athayahisyam/dcdshit/assets/31662860/aa387d22-16af-44d9-8f07-64ca4c90e223)

Pada tahapan ini outlier diketahui dengan Box Plot. Adapun outlier yang ada tidak diubah, karena algoritma *Random Forest* andal mengatasi outlier ini. Pada algoritma *random forest*, *outlier* ditangani dengan tiga cara:

* Struktur Pohon: struktur pohon tidak dipengaruhi oleh *outlier* karena pemecahan pohon dilakukan berdasarkan *threshold* dan tidak bergantung pada jarak pengukuran
* Pemerataan: Prediksi akhir dilakukan pemerataan melalui *voting*, dimana mekanisme *vote* tidak meluluskan *outlier*. Pemerataan juga "melarutkan" pengaruh nilai *outlier*.
* *Bootsrapping*: Penggunaan metode *bootstrapping* mengurangi pengaruh *outlier* pada model final.

Pada Gambar 2, dapat diperhatikan bahwa `Solids` memiliki karakteristik 

## Modeling

Pada kasus projek ini, Algoritma *Random Forest* bekerja dengan beberapa tahapan, yaitu:

1. *Ensemble of Decision Trees*: Algoritma *Random Forest* terdiri dari ansambel pohon keputusan. Setiap pohon keputusan dibangun secara independen dengan memilih subset data pelatihan dan subset fitur input secara acak.

2. Proses Pelatihan: Selama proses pelatihan, setiap pohon keputusan pada *Random Forest* belajar untuk mengklasifikasikan sampel berdasarkan subset fitur input yang berbeda. Keacakan dalam pemilihan fitur dan pengambilan sampel data ini membantu mengurangi *overfitting* dan meningkatkan keragaman di antara masing-masing pohon.

3. Mekanisme *Voting*: Setelah pohon keputusan dilatih, mereka bekerja sama untuk membuat prediksi. Ketika sampel air baru ditambahkan ke model, setiap pohon keputusan mengklasifikasikan sampel secara independen berdasarkan serangkaian fitur yang dipilih. Klasifikasi akhir ditentukan melalui mekanisme pemungutan suara, di mana setiap prediksi pohon keputusan dihitung sebagai *vote*.

4. Agregasi Prediksi: Prediksi dari semua pohon keputusan digabungkan, dan *vote* mayoritas atau rata-rata (dalam kasus regresi) diambil sebagai prediksi akhir. Misalnya, dalam masalah klasifikasi biner seperti daya minum air (0 atau 1), prediksi akhir dapat ditentukan oleh *vote* mayoritas dari pohon keputusan.

5. Menangani Kelas yang Tidak Seimbang: *Random Forest* dapat menangani kelas yang tidak seimbang, seperti dalam kasus air yang dapat diminum di mana jumlah sampel yang dapat diminum dan yang tidak dapat diminum mungkin tidak sama. Mekanisme pemilihan pada algoritma *Random Forest* memperhitungkan prediksi dari banyak pohon, yang membantu membuat prediksi yang lebih akurat untuk kedua kelas, bahkan jika datanya tidak seimbang.

6. Feature Importance: Algoritma *Random Forest* juga memberikan informasi tentang pentingnya setiap fitur masukan. Model mengukur kontribusi tiap fitur pada ansambel pohon keputusan untuk klasifikasi. Informasi ini dapat berharga untuk memahami pentingnya fitur yang berbeda dalam menentukan daya minum air.

Adapun tahapan yang dilakukan dalam melakukan training adalah sebagai berikut:

1. Pemecahan dataset menjadi fitur (X) dan target variabel y, yakni `Potability`

2. Pemecahan dua variabel tersebut menjadi variabel *train* dan *test*, `random state` yang ditentukan adalah 52. Sebanyak 20% data digunakan untuk *test* dan sisanya, sebanyak 80% digunakan untuk *training.*

3. Karena dataset terdiri dari hasil pengukuran yang berbeda metrik, dilakukan tahapan *feature scaling*. Dari hasil *feature scaling*, data berhasil diturunkan dengan menjaga variabilitasnya

4. Melatih model dengan Algoritma Random Forest, dengan parameter` *random state`yang ditentukan adalah 52 dan nilai `n_estimators` adalah *default* 300.

## Evaluation

Dari hasil prediksi, ditemukan bahwa akurasi model mencapai 68%. Adapun metrik evaluasi adalah sebagai berikut:

* Akurasi: keakuratan total dari prediksi model dengan membandingkan label hasil prediksi dan label pada dokumen.
* *Confusion matrix*: Matriks yang memetakan performa model dengan mempresentasikan jumlah *True Positive, True Negative, False Negative* dan *False Positive*
* Presisi: Presisi mengukur proporsi antara sampel yang secara akurat diprediksi oleh model dengan seluruh sampel yang dilabeli "positif" oleh model.
* *Recall*: Ukuran proporsi antara sampel yang dilabeli "positif" oleh model dan dengan sampel yang berlabel asli "positif".
* Skor F1: Rata-rata harmonik antara presisi dan *recall*. 
* *Support*: Jumlah sampel pada tiap kelas.
  
Dari laporan klasifikasi dapat dilihat:

- Akurasi model 67.68% ini berarti 67.68% dari sampel air pada dataset test diklasifikasikan secara akurat baik antara bisa diminum (1) atau tidak bisa diminum (0)
- Pada confusion matrix terdapat 4 nilai, nilai pertama adalah 356, yang merupakan True Positives, bermakna model mengidentifikasikan secara tepat 356 sampel sebagai air dapat diminum.
- Kedua, nilai True Negative 88, bermakna model secara akurat mengidentifikasi 88 sampel sebagai air tidak dapat diminum.
- Ketiga, nilai False Negative 46, yang bermakna model melakukan kesalahan klasifikasi 46 sampel air yang aslinya dapat diminum, menjadi diklasifikasikan sebagai air tidak dapat diminum.
- Keempat, nilai False Positive 166, berarti model melakukan kesalahan klasifikasi 166 sampel yang tidak bisa diminum, dan mengklasifikasikannya sebagai air yang bisa diminum.
- Pada data yang dilabeli 0/tidak dapat diminum:
	- Presisi: Presisi 0,68 menunjukkan bahwa dari semua sampel yang diprediksi sebagai "tidak dapat diminum", 68% diklasifikasikan dengan benar. Hal ini menunjukkan bahwa model tersebut memiliki tingkat akurasi yang sedang dalam mengidentifikasi sampel air yang “tidak dapat diminum”.
	- *Recall*: Skor 0,89 berarti bahwa model mengidentifikasi dengan benar 89% dari sampel "yang tidak dapat diminum" yang sebenarnya. Ini menunjukkan model memiliki *recall* yang tinggi terhadap kelas ini, menunjukkan bahwa model memiliki kemampuan yang baik untuk menangkap sebagian besar sampel air yang "tidak dapat diminum".
	- Skor F1: Skor F1 sebesar 0,77 menggabungkan presisi dan *recall*, memberikan ukuran keseluruhan performa model untuk kelas "tidak dapat diminum". Ini menunjukkan keseimbangan antara presisi dan ingatan, dengan mempertimbangkan *false positive* dan *false negative*.
	- *Support*: Nilai 402 menunjukkan jumlah sampel di kelas "tidak dapat diminum".
- Pada data yang dilabeli 1/dapat diminum
	- Presisi: Presisi 0,66 menunjukkan bahwa 66% sampel yang diprediksi sebagai "dapat diminum" diklasifikasikan dengan benar. Meskipun tidak setinggi yang diinginkan, ia masih menangkap sebagian besar sampel air yang "dapat diminum".
	- *Recall*: Skor *Recall* 0,35 menunjukkan bahwa model hanya mampu mengidentifikasi 35% dari sampel "dapat diminum" yang sebenarnya. Hal ini menunjukkan bahwa model tersebut memiliki kemampuan yang lebih rendah untuk mengidentifikasi dengan benar kelas "dapat diminum", yang berpotensi menghasilkan lebih banyak *false negative*.
	- Skor F1: Skor F1 0,45 relatif lebih rendah, yang mencerminkan *trade-off* antara presisi dan *recall* untuk kelas "dapat diminum". Ini menunjukkan bahwa kinerja model untuk kelas ini tidak seimbang dengan kelas "tidak dapat diminum".
	- *Support*: Nilai dukungan 254 mewakili jumlah sampel di kelas "dapat diminum".

Sebagai usaha meningkatkan akurasi pada model, dilakukan upaya *feature selection* untuk memilih fitur-fitur yang paling berpengaruh terhadap fitur `potability` pada dataset. Dari hasil usaha *tuning* menggunakan seleksi fitur, tidak ditemukan peningkatan yang signifikan dari akurasi model. 

# Kesimpulan 

Dari hasil evaluasi model, ditemukan bahwa akurasi model hanya berada pada kisaran 68%, meskipun telah dilakukan *feature selection tuning*. Hal ini mungkin terjadi karena:

- Jumlah data yang kurang
- Algoritma yang digunakan tidak tepat.

Namun perlu digaris bawahi bahwa model ini mampu mengidentifikasi hampir 68% secara akurat antara air yang bisa diminum dan tidak diminum. Saran untuk penelitian/projek selanjutnya adalah menggunakan algoritma lain, semisal yang digunakan oleh Gao, et al. [3] yaitu CNN untuk melakukan identifikasi lebih lanjut. 

# Referensi

[1] Roberto F., et al. Evaluation of a GFP reporter gene construct for environmental arsenic detection. Talanta, 2002, 58(1): 181-188.
[2] Erdogan O., et al. Critical evaluation of wastewater treatment and disposal strategies for Istanbul with regards to water quality monitoring study results. ELSEVISE, 2008, 226: 231-248.
[3] Gao, et al. Water Potability Analysis and Prediction, Highlight in Science, Engineering and Technology, Vol. 6, AMMSAC 2022


**---Ini adalah bagian akhir laporan---**
