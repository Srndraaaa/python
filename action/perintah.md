saya mengikuti sebuah perlombaan data mining, dengan kriteria sebagai berikut :

1.overview :
Selamat datang di ACTION 2025 Cabang Data Mining! Pada kompetisi ini, peserta akan menghadapi tantangan label discovery pada kumpulan gambar makanan tradisional Indonesia.

Pesatnya pertumbuhan volume data citra digital di berbagai domain mulai dari media sosial hingga basis data ilmiah telah mendorong munculnya disiplin ilmu Data Mining Citra (Image Mining). Bidang ini bertujuan untuk mengekstrak pengetahuan, pola, dan hubungan tersembunyi dari koleksi citra yang masif. Namun, salah satu hambatan terbesar dalam memproses kumpulan data citra besar adalah masalah anotasi (pelabelan). Secara tradisional, citra memerlukan label semantik (konsep tingkat tinggi, seperti 'mobil' atau 'pantai') agar dapat digunakan dalam model pembelajaran terawasi (supervised learning). Proses pelabelan manual sangat memakan waktu, mahal, dan tidak praktis untuk skala data besar.

Oleh karena itu, konsep Label Discovery menjadi esensial. Label discovery adalah proses krusial dalam Data Mining Citra yang berfokus pada mengidentifikasi atau menemukan label semantik yang paling relevan dan bermakna pada citra yang tidak berlabel. Proses ini sering melibatkan teknik pembelajaran tanpa pengawasan (unsupervised learning) seperti clustering untuk mengelompokkan citra berdasarkan fitur visual yang serupa, memungkinkan penemuan konsep kategoris secara semi-otomatis dari data mentah.

Dalam konteks kompetisi ini, ACTION 2025 secara spesifik memberikan tantangan kepada peserta untuk menerapkan praktik Label Discovery pada kumpulan citra makanan tradisional Indonesia. Peserta diharapkan mampu menemukan dan menetapkan (discover and assign) label semantik berupa 15 kategori makanan tradisional pada citra yang tidak berlabel. Tantangan ini menjadi langkah penting sebelum melatih model untuk memprediksi data uji, sekaligus mengasah kemampuan teknis peserta dalam menjembatani kesenjangan antara representasi piksel mentah dan konsep kuliner tingkat tinggi yang bermakna.

2.Description :
Anda akan menerima kumpulan gambar tanpa label, namun sudah mengetahui daftar 15 kemungkinan kategori makanan Nusantara yang terdapat dalam data, Daftar label sebagai berikut :

1. Ayam Bakar
2. Ayam Betutu
3. Ayam Goreng
4. Ayam Pop
5. Bakso
6. Coto Makassar
7. Gado Gado
8. Gudeg
9. Nasi Goreng
10. Pempek
11. Rawon
12. Rendang
13. Sate Madura
14. Sate Padang
15. Soto

Tugas Anda adalah menemukan label yang paling sesuai untuk setiap gambar pada data latih (train), kemudian melatih model terbaik untuk memprediksi data uji (test).

3.Evaluasi dilakukan menggunakan akurasi klasifikasi sebagai metrik utama:
```python
from sklearn.metrics import accuracy_score
score = accuracy_score(solution, submission)
Semua prediksi harus mengikuti format yang sesuai dengan file test.csv.
```
note :
saya sudah mengirimkan buku panduan dari lomba tersebut, anda harus baca untuk memberikan solusi dari permasalahan.

peraturan yang harus diikuti :
1. gunakan ilmu-ilmu nyata data science dalam memberikan solusi dan output
2. baca seluruh buku panduan sampai anda memahaminya dan memahami struktur permasalahan
3. model harus memiliki akurasi sebesar >0,90 (diuji menggunakan metode evaluasi yang sudah ditentukan)
4. anda tidak perlu install library-library yangg dibutuhkan (cukup tuliskan import seluruh library di awal).

output yang diharapkan :
1. berikan model machine learning untuk menyelesaikan permasalahan (secara bertahap dari awal hingga ahir)
2. codingan anda harus memiliki struktur penulisan seperti manusia (jangan terlihat seperti AI)
3. gunakan library machine learning (compare antara model 1 dan model lainnya)
4. struktur submission.csv harus memiliki struktur yang sama seperti yang sudah ditentukan