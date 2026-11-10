# KOMPILASI LOG INTERAKSI LLM — KELOMPOK GEN-AI-AJA

| Parameter | Keterangan |
| :--- | :--- |
| **Proyek** | Conditional GAN (cGAN) Klasik pada Dataset MNIST (PyTorch) |
| **Mata Kuliah** | Generative AI (CAK4KBB3) |
| **Metodologi** | *Vibe Coding from Scratch* bersama Large Language Model (LLM) |
| **Penyusun** | Muhammad Furqan Maulidi (Anggota 4 - Lead Documentarian & Debugger) |
| **Status Dokumen** | Final — Kompilasi Lengkap Anggota 1 s.d. 4 |

---

## 👥 1. Identitas Tim & Distribusi Peran LLM

| No | Nama Anggota | NIM | Peran Utama | Fokus Interaksi LLM & Cakupan Modul |
| :-: | :--- | :-: | :--- | :--- |
| 1 | **Fadhil Abithyasa Effendi** | 103012330370 | Prompt Engineer & LLM Navigator | Arsitektur Model cGAN (`models.py`), Kontrak Kode (`config.py`), Inferensi (`generate.py`), Sanity Check (`verify_integration.py`). |
| 2 | **Muhammad Nazriel Ihram** | 103012300269 | Data Pipeline Engineer | Pipeline Dataset MNIST (`data.py`), Normalisasi `[-1, 1]`, DataLoader, & Kontrol Reprodusibilitas Seed. |
| 3 | **Damar Muharram** | 103012300096 | Training & Tuning Engineer | Adversarial Training Loop (`train.py`), Loss Functions, Utilitas Visualisasi (`utils.py`), & Eksperimen Tuning (Label Smoothing & TTUR). |
| 4 | **Muhammad Furqan Maulidi** | 103012330213 | Lead Documentarian & Debugger | Penanganan Eror Compiler/Sistem, Migrasi Colab, Rekam Jejak Debugging (`log_debugging.md`), `README.md`, & Laporan Akhir (`laporan_akhir.md`). |

---

## 📝 2. Ringkasan Log Interaksi per Anggota

### 2.1 Anggota 1 — Arsitektur & Kontrak Kode (`models.py`, `config.py`)
- **Fokus Prompting:** Merancang arsitektur Generator dan Discriminator cGAN yang terikat pada pengondisian label digit MNIST (0–9).
- **Hasil Kolaborasi LLM:**
  1. Penetapan `config.py` sebagai sumber kebenaran tunggal (*Single Source of Truth*) untuk hiperparameter (`z_dim=100`, `num_classes=10`, `lr=0.0002`, `batch_size=64`).
  2. Implementasi Generator dengan penggabungan tensor laten z dan *label embedding* `[B, 150]`, diproyeksikan ke matriks citra 28x28 via `ConvTranspose2d` dan aktivasi `Tanh`.
  3. Implementasi Discriminator dengan penggabungan peta spasial label `[B, 2, 28, 28]` via `Conv2d`, `LeakyReLU(0.2)`, `Dropout2d(0.25)`, dan output logit skalar.

### 2.2 Anggota 2 — Pipeline Data & Preprocessing (`data.py`)
- **Fokus Prompting:** Membangun *Data Pipeline* MNIST yang efisien dan stabil terhadap masalah statistik batch.
- **Hasil Kolaborasi LLM:**
  1. Penggunaan `transforms.Normalize((0.5,), (0.5,))` untuk menskalakan piksel `[0, 1]` ke rentang `[-1, 1]` agar selaras dengan output `Tanh` Generator.
  2. Penambahan parameter `drop_last=True` pada `DataLoader` untuk mencegah eror kalkulasi `BatchNorm1d` ketika *batch* terakhir hanya menyisa 1 sampel data.
  3. Fungsi *seed setter* universal untuk memastikan reprodusibilitas inisialisasi bobot di PyTorch dan NumPy.

### 2.3 Anggota 3 — Training Loop & Hyperparameter Tuning (`train.py`, `utils.py`)
- **Fokus Prompting:** Menyusun loop pelatihan adversarial cGAN serta mengimplementasikan variasi argumen CLI untuk eksperimen pencapaian ekuilibrium.
- **Hasil Kolaborasi LLM:**
  1. Penerapan fungsi kerugian `BCEWithLogitsLoss` untuk stabilitas kalkulasi numerik gradien.
  2. Fitur pencatatan metrik otomatis ke file CSV (`outputs/logs/metrics_*.csv`) dan penyimpanan grid citra sampel per epoch (`outputs/samples/`).
  3. Implementasi modul penyeimbang pelatihan: *One-Sided Label Smoothing* (target real = 0.9) dan *Two Time-Scale Update Rule* (TTUR dengan LR_G = 0.0002, LR_D = 0.0001).

### 2.4 Anggota 4 — Debugging, Eksekusi Colab & Dokumentasi (`docs/`)
- **Fokus Prompting:** Mengatasi pemblokiran biner/DLL oleh sistem keamanan lokal (AppLocker), migrasi eksekusi ke Google Colab, serta pengompilasian laporan.
- **Hasil Kolaborasi LLM:**
  1. Formulasi prompt perbaikan eror `AppLocker / DLL load failed` hingga diputuskan migrasi eksekusi ke Google Colab dengan akselerasi T4 GPU.
  2. Pembaruan aturan `.gitignore` menjadi `outputs/checkpoints/**/*.pth` untuk mencegah file bobot besar ter-commit.
  3. Penyusunan `docs/log_debugging.md` v2.0, pembaruan `README.md`, dan penyusunan `docs/laporan_akhir.md` Bab 1–5 secara terstruktur.

---

## 📂 3. Direktori Riwayat Chat LLM (`docs/chats/`)

Seluruh arsip bukti obrolan dan tautan interaksi LLM disimpan pada direktori `docs/chats/` dengan struktur berikut:

```text
docs/chats/
├── link_obrolan.md           # Daftar tautan (share link) obrolan LLM Anggota 1-4
├── anggota1_arch.md          # Arsip chat Anggota 1 (Arsitektur cGAN & Config)
├── anggota2_data.md          # Arsip chat Anggota 2 (Data Pipeline & Seed)
├── anggota3_train.md         # Arsip chat Anggota 3 (Training Loop & Tuning)
└── anggota4_debug_doc/       # Arsip chat Anggota 4 (Debugging & Dokumentasi)
