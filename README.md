# Conditional GAN (cGAN) Klasik pada Dataset MNIST — From Scratch

Proyek ini dibangun sebagai pemenuhan Tugas Besar / Ujian Tengah Semester (UTS) mata kuliah **Generative AI (CAK4KBB3)**. 
Model dikembangkan menggunakan paradigma **Vibe Coding from scratch**, berkolaborasi aktif bersama Large Language Model (LLM) tanpa menyalin (*copy-paste*) kode utuh dari repositori atau tutorial internet.

---

## 👥 Tim Pengembang (Kelompok GEN-AI-AJA)

| Peran | Nama Anggota | NIM | Fokus & Tanggung Jawab |
| :--- | :--- | :--- | :--- |
| **Anggota 1** | **Dil** | [NIM-1] | Prompt Engineer & LLM Navigator (Arsitektur Model cGAN & Kontrak Kode) |
| **Anggota 2** | **MUHAMMAD NAZRIEL IHRAM** | 103012300269 | Data Pipeline & Preprocessing Engineer (Dataset MNIST & DataLoader) |
| **Anggota 3** | [Nama Anggota 3] | [NIM-3] | Training Loop & Hyperparameter Tuning Engineer (Loss & Optimasi) |
| **Anggota 4** | [Nama Anggota 4] | [NIM-4] | Lead Documentarian & Error Handler (Log Interaksi, Debugging, & Laporan) |

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
python train.py
```

### 5. Generate Angka 0–9
```bash
python generate.py
```

---

## 📜 Log Interaksi & Debugging LLM
Riwayat interaksi dengan LLM, proses penanganan eror mandiri, serta tautan share chat dapat dilihat secara transparan pada folder [docs/](docs/).
