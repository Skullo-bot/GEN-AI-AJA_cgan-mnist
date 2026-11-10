# Conditional GAN (cGAN) Klasik pada Dataset MNIST — From Scratch

Proyek ini dibangun sebagai pemenuhan Tugas Besar / Ujian Tengah Semester (UTS) mata kuliah **Generative AI (CAK4KBB3)**. 
Model dikembangkan menggunakan paradigma **Vibe Coding from scratch**, berkolaborasi aktif bersama Large Language Model (LLM) tanpa menyalin (*copy-paste*) kode utuh dari repositori atau tutorial internet.

---

## 👥 Tim Pengembang (Kelompok GEN-AI-AJA)

| Peran | Nama Anggota | NIM | Fokus & Tanggung Jawab |
| :--- | :--- | :--- | :--- |
| **Anggota 1** | **Fadhil Abithyasa Effendi** | 103012330370 | Prompt Engineer & LLM Navigator (Arsitektur Model cGAN & Kontrak Kode) |
| **Anggota 2** | **Muhammad Nazriel Ihram** | 103012300269 | Data Pipeline & Preprocessing Engineer (Dataset MNIST & DataLoader) |
| **Anggota 3** | **Damar Muharram** | 103012300096 | Training Loop & Hyperparameter Tuning Engineer (Loss & Optimasi) |
| **Anggota 4** | **Muhammad Furqan Maulidi** | 103012330213 | Lead Documentarian & Error Handler (Log Interaksi, Debugging, & Laporan) |
---

## 🧠 Konsep & Arsitektur Model

Conditional Generative Adversarial Network (cGAN) mengondisikan proses pembentukan citra dengan menambahkan informasi label kelas (digit 0–9).

- **Dataset:** MNIST (60.000 citra tulisan tangan angka 0–9, 1 channel grayscale, resolusi 28×28 piksel, dinormalisasi ke rentang `[-1, 1]`).
- **Generator:**
  - Input: Vektor laten (*noise*) $z \in \mathbb{R}^{100}$ digabungkan dengan tensor embedding label kelas.
  - Lapisan: Proyeksi Linear / Transposed Convolution (`ConvTranspose2d`) + `BatchNorm2d` + `ReLU`.
  - Output: Citra sintetis `[B, 1, 28, 28]` dengan fungsi aktivasi `Tanh`.
- **Discriminator:**
  - Input: Citra `[B, 1, 28, 28]` digabungkan dengan representasi spasial embedding label `[B, 1, 28, 28]` menjadi tensor `[B, 2, 28, 28]`.
  - Lapisan: `Conv2d` + `LeakyReLU` + `Dropout`.
  - Output: Logit skalar real/fake (menggunakan `BCEWithLogitsLoss`).

---

## 📁 Struktur Repositori

```text
cgan-mnist/
│
├── .gitignore                      # Aturan filter file Git
├── requirements.txt                # Daftar pustaka dependensi proyek
├── README.md                       # Ringkasan identitas proyek & panduan
│
├── config.py                       # Kontrak variabel & hyperparameter global
├── models.py                       # Arsitektur Generator & Discriminator cGAN
├── data.py                         # Pipeline dataset MNIST & DataLoader
├── train.py                        # Training loop & optimasi model
├── utils.py                        # Utilitas plotting loss, visualisasi sampel, & seed
├── generate.py                     # Skrip inferensi angka 0–9 dari checkpoint
├── verify_integration.py           # Skrip uji konsistensi tensor shape antar modul
│
├── dataset/                        # Lokasi unduhan dataset MNIST (diabaikan Git)
│
├── docs/                           # Pusat dokumentasi tim
│   ├── system_prompt.md            # Arsip system prompt awal anggota
│   ├── log_interaksi.md            # Kompilasi Log Interaksi LLM
│   ├── log_debugging.md            # Rekam jejak compiler error & analisis solusi
│   ├── laporan_akhir.md            # Laporan akhir komprehensif
│   └── chats/                      # Bukti riwayat chat per anggota
│
└── outputs/                        # Luaran eksperimen
    ├── checkpoints/                # Bobot model tersimpan (.pth)
    ├── samples/                    # Grid citra angka hasil generate per epoch
    ├── plots/                      # Grafik kurva loss D vs G
    └── logs/                       # File riwayat metrik (CSV)
```

---

## 🚀 Panduan Menjalankan Proyek

### 1. Kloning Repositori & Persiapan Lingkungan
```bash
git clone https://github.com/Skullo-bot/GEN-AI-AJA_cgan-mnist.git
cd GEN-AI-AJA_cgan-mnist
python -m venv .venv
# Mengaktifkan Virtual Environment di Windows PowerShell:
.venv\Scripts\Activate.ps1
```

### 2. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 3. Uji Konsistensi Modul (Sanity Check)
```bash
python verify_integration.py
```

### 4. Pelatihan Model (Training)
```bash
python train.py --exp_name baseline
python train.py --exp_name smooth --label_smoothing --compare_with baseline
python train.py --exp_name ttur --lr_g 0.0002 --lr_d 0.0001 --compare_with baseline
python train.py --exp_name smooth_ttur --label_smoothing --lr_g 0.0002 --lr_d 0.0001 --compare_with baseline smooth ttur
```

### 5. Generate Angka 0–9
```bash
python generate.py --checkpoint outputs/checkpoints/smooth_ttur/best.pth --digit 7
```
---

## 📊 Hasil Eksperimen & Hyperparameter Tuning

| Eksperimen | Learning Rate (G / D) | Label Smoothing | d_loss (Epoch 30) | g_loss (Epoch 30) | Analisis Singkat |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | **0.0002 / 0.0002** | Tidak | 0.2323 | 3.1047 | Discriminator dominan G loss terus membengkak |
| **Smooth** | **0.0002 / 0.0002** | Ya | 0.3462 | 2.6766 | Menahan Discriminator agar tidak terlalu yakin (overconfident) |
| **TTUR** | **0.0002 / 0.0001** | Tidak | 0.3783 | 1.9799 | Menyeimbangkan ritme belajar g_loss turun signifikan (−36.2%) |
| **Smooth + TTUR** | **0.0002 / 0.0001** | Ya | 0.4427 | 1.9862 | Performa paling stabil mendekati ekuilibrium ideal (ln2≈0.693) |

## 📜 Log Interaksi & Debugging LLM
Riwayat interaksi dengan LLM, proses penanganan eror mandiri, serta tautan share chat dapat dilihat secara transparan pada folder [docs/](docs/).
