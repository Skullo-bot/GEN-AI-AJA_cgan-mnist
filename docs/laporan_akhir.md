# LAPORAN AKHIR PROYEK CONDITIONAL GAN (cGAN) PADA DATASET MNIST

| Parameter               | Spesifikasi / Keterangan                                      |
| :---------------------- | :------------------------------------------------------------ |
| **Mata Kuliah**         | Generative AI (CAK4KBB3)                                      |
| **Topik Proyek**        | Conditional GAN (cGAN) Klasik pada Dataset MNIST (PyTorch)    |
| **Metode Pengembangan** | _Vibe Coding from Scratch_ bersama Large Language Model (LLM) |
| **Repositori GitHub**   | https://github.com/Skullo-bot/GEN-AI-AJA_cgan-mnist           |
| **Kelompok**            | GEN-AI-AJA                                                    |

---

## 👥 Tim Pengembang & Pembagian Tugas

| No  | Nama Anggota                 |     NIM      | Peran Utama                   | Deskripsi Kontribusi & Tanggung Jawab                                                                                                 |
| :-: | :--------------------------- | :----------: | :---------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
|  1  | **Fadhil Abithyasa Effendi** | 103012330370 | Ketua & Prompt Engineer       | Merancang arsitektur cGAN (`models.py`), menentukan kontrak `config.py`, dan menyusun skrip inferensi `generate.py`.                  |
|  2  | **Muhammad Nazriel Ihram**   | 103012300269 | Data Pipeline Engineer        | Membangun pipeline dataset MNIST (`data.py`), penanganan normalisasi `[-1, 1]`, dan kontrol reprodusibilitas _seed_.                  |
|  3  | **Damar Muharram**           | 103012300096 | Training & Tuning Engineer    | Mengimplementasikan _training loop_ adversarial (`train.py`), fungsi utilitas (`utils.py`), serta eksperimen hyperparameter.          |
|  4  | **Muhammad Furqan Maulidi**  | 103012330213 | Lead Documentarian & Debugger | Mengelola rekam jejak eror (`log_debugging.md`), kompilasi interaksi (`log_interaksi.md`), penyusunan `README.md`, dan laporan akhir. |

---

## BAB 1: PENDAHULUAN

### 1.1 Latar Belakang

Generative Adversarial Network (GAN) konvensional mampu membangkitkan citra sintetis dari distribusi vektor laten acak ($z$). Namun, GAN klasik memiliki keterbatasan dalam mengontrol kelas atau identitas citra yang dihasilkan. Conditional GAN (cGAN) mengatasi kendala ini dengan mengondisikan Generator ($G$) dan Discriminator ($D$) menggunakan informasi tambahan berupa label kelas ($y$).

Pada proyek ini, model cGAN dibangun secara khusus untuk membangkitkan citra sintetis angka tulisan tangan (digit 0–9) dari dataset MNIST. Seluruh kode dikembangkan dari awal (_from scratch_) tanpa bergantung pada kerangka kerja generatif tingkat tinggi, memanfaatkan pendekatan _Vibe Coding_ yang dikawal secara ketat oleh pengujian unit dan dokumentasi LLM.

### 1.2 Tujuan Proyek

1. Merancang dan mengimplementasikan arsitektur cGAN berbasis PyTorch secara modular terikat kontrak `config.py`.
2. Mengatasi fenomena _Discriminator Dominance_ dan _Intra-class Mode Collapse_ melalui eksperimen _Hyperparameter Tuning_ (Label Smoothing dan Two Time-Scale Update Rule).
3. Mendokumentasikan seluruh proses _prompt engineering_, penanganan eror compiler, dan analisis dinamika pelatihan secara transparan.

---

## BAB 2: ARSITEKTUR MODEL & PIPELINE DATA

### 2.1 Pipeline Data (`data.py`)

- **Dataset:** MNIST (60.000 data latih, 10.000 data uji, resolusi 28×28 piksel grayscale).
- **Transformasi:** Peta intensitas piksel `[0, 1]` dinormalisasi ke rentang `[-1, 1]` menggunakan `transforms.Normalize((0.5,), (0.5,))`, selaras dengan fungsi aktivasi `Tanh` pada Generator.
- **DataLoader:** Menggunakan `batch_size=64` dengan `drop_last=True` pada data latih untuk mencegah eror kalkulasi statistik pada lapisan `BatchNorm1d` ketika batch terakhir hanya berisi 1 sampel.

### 2.2 Arsitektur Generator (`models.py`)

- **Input:** Vektor laten acak $z \in \mathbb{R}^{100}$ dan label kelas $y \in \{0..9\}$.
- **Pengondisian Label:** Label di-embed menggunakan `nn.Embedding(10, 50)` lalu di-konkatenasi dengan $z$ menjadi tensor `[B, 150]`.
- **Lapisan:** `Linear` projection $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ Reshape `[B, 128, 7, 7]` $\rightarrow$ `ConvTranspose2d` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `ConvTranspose2d` $\rightarrow$ `Tanh`.
- **Output:** Tensor citra sintetis `[B, 1, 28, 28]`.

### 2.3 Arsitektur Discriminator (`models.py`)

- **Input:** Tensor citra `[B, 1, 28, 28]` dan label $y$.
- **Pengondisian Label:** Label di-embed menjadi peta spasial `[B, 1, 28, 28]` via `nn.Embedding(10, 784)` lalu di-konkatenasi pada channel citra menjadi tensor `[B, 2, 28, 28]`.
- **Lapisan:** `Conv2d` $\rightarrow$ `LeakyReLU(0.2)` $\rightarrow$ `Conv2d` $\rightarrow$ `LeakyReLU(0.2)` $\rightarrow$ `Dropout2d(0.25)` $\rightarrow$ `Flatten` $\rightarrow$ `Linear(6272, 1)`.
- **Output:** Logit skalar real/fake yang dievaluasi dengan `BCEWithLogitsLoss`.

---

## BAB 3: METODOLOGI PELATIHAN & EXPERIMENTAL DESIGN

Pelatihan dijalankan selama 30 epoch pada setiap eksperimen menggunakan optimizer Adam (`beta1 = 0.5`, `beta2 = 0.999`) dan fungsi loss `BCEWithLogitsLoss`. Nilai loss Discriminator dihitung dari rata-rata loss citra asli dan citra palsu, yaitu `L_D = 0.5 * (L_real + L_fake)`, dengan titik keseimbangan ideal berada di kisaran `ln(2) ≈ 0.693`.

Untuk mencari keseimbangan terbaik antara Generator dan Discriminator, dirancang 4 variasi eksperimen:

1. **Baseline (`baseline`):** Learning rate seragam $LR_G = 0.0002, LR_D = 0.0002$ tanpa smoothing.
2. **Label Smoothing (`smooth`):** Menerapkan _one-sided label smoothing_ (target real $D$ diturunkan dari $1.0$ ke $0.9$).
3. **Two Time-Scale Update Rule (`ttur`):** Membedakan rasio pembelajaran ($LR_G = 0.0002, LR_D = 0.0001$).
4. **Kombinasi (`smooth_ttur`):** Menggabungkan _Label Smoothing_ (0.9) dan TTUR ($LR_G = 0.0002, LR_D = 0.0001$).

---

## BAB 4: HASIL DAN ANALISIS

### 4.1 Perbandingan Metrik Kuantitatif (30 Epoch)

| Eksperimen        | $LR_G$ | $LR_D$ | Label Smooth | $D\_Loss$ (Ep 1) | $G\_Loss$ (Ep 1) | $D\_Loss$ (Ep 30) | $G\_Loss$ (Ep 30) | Catatan Dinamika                            |
| :---------------- | :----: | :----: | :----------: | :--------------: | :--------------: | :---------------: | :---------------: | :------------------------------------------ |
| **Baseline**      | 0.0002 | 0.0002 |   Nonaktif   |      0.3886      |      1.8626      |      0.2323       |      3.1047       | $D$ terlalu dominan; $G\_loss$ membengkak.  |
| **Smooth**        | 0.0002 | 0.0002 | Aktif (0.9)  |      0.4535      |      1.9228      |      0.3462       |      2.6766       | Menahan _overconfidence_ $D$.               |
| **TTUR**          | 0.0002 | 0.0001 |   Nonaktif   |      0.5585      |      1.1683      |      0.3783       |      1.9799       | $G\_loss$ turun signifikan (−36.2%).        |
| **Smooth + TTUR** | 0.0002 | 0.0001 | Aktif (0.9)  |      0.5822      |      1.3003      |    **0.4427**     |    **1.9862**     | **Kondisi paling stabil ($\approx \ln 2$)** |

### 4.2 Pembahasan Dinamika Pelatihan

1. **Kegagalan Baseline:** Pada eksperimen baseline, Discriminator belajar terlalu cepat dibanding Generator ($D\_loss$ turun ke 0.2323 pada epoch 30). Hal ini menyebabkan gradien yang dikirim ke Generator menjadi kurang informatif, memicu _g_loss_ membengkak ke 3.1047 serta penurunan keragaman sampel dalam satu kelas (_intra-class mode collapse_).
2. **Efektivitas TTUR & Smoothing:** Penurunan $LR_D$ menjadi 0.0001 memberi kesempatan bagi Generator untuk menyesuaikan bobot secara efektif. Kombinasi TTUR dan Label Smoothing pada `smooth_ttur` berhasil menjaga $D\_loss$ di angka 0.4427 (mendekati 0.693) dan menstabilkan $G\_loss$ di angka 1.9862, menghasilkan struktur angka yang jelas dan bervariasi.

---

## BAB 5: KESIMPULAN

1. Model Conditional GAN (cGAN) untuk pembentukan citra sintetis angka MNIST berhasil dikembangkan _from scratch_ dan terintegrasi secara modular.
2. Penerapan kombinasi **Two Time-Scale Update Rule (TTUR)** dan **One-Sided Label Smoothing** terbukti secara kuantitatif mampu mengatasi dominasi Discriminator serta menstabilkan kurva _loss_ Generator pada nilai ideal.
3. Dokumentasi eksperimen, rekam jejak debugging, dan riwayat interaksi LLM telah tersusun secara transparan pada repositori proyek.
