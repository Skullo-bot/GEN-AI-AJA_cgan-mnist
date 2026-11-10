# RIWAYAT OBROLAN & LOG INTERAKSI LLM — ANGGOTA 4 (LEAD DOCUMENTARIAN & ERROR HANDLER)

**Identitas Dokumen:**
- **Kelompok:** GEN-AI-AJA
- **Mata Kuliah:** Generative AI (CAK4KBB3)
- **Topik Proyek:** Conditional GAN Klasik untuk MNIST (angka tulisan tangan 0–9)
- **Anggota:** Anggota 4 — Muhammad Furqan Maulidi (NIM: 103012330213)
- **Peran:** Lead Documentarian & Error Handler
- **Target Modul:** `docs/log_interaksi.md`, `docs/log_debugging.md`, `README.md`, `docs/laporan_akhir.md`, `.gitignore`, `outputs/`
- **Rentang Sesi:** 07/11/2026 – 10/11/2026

---

## DAFTAR ISI KRONOLOGI SESI
1. [Sesi 1: Inisiasi Log Interaksi & Formulasi Template Log LLM (`docs/log_interaksi.md`)](#sesi-1-inisiasi-log-interaksi--formulasi-template-log-llm-docslog_interaksimd)
2. [Sesi 2: Dokumentasi Debugging Awal & Isolasi Eror GUI Git (`docs/log_debugging.md` v1.0)](#sesi-2-dokumentasi-debugging-awal--isolasi-eror-gui-git-docslog_debuggingmd-v10)
3. [Sesi 3: Penanganan AppLocker & Pemblokiran DLL PyTorch Lokal (DBG-04)](#sesi-3-penanganan-applocker--pemblokiran-dll-pytorch-lokal-dbg-04)
4. [Sesi 4: Formulasi Strategi Migrasi Eksekusi ke Google Colab (T4 GPU)](#sesi-4-formulasi-strategi-migrasi-eksekusi-ke-google-colab-t4-gpu)
5. [Sesi 5: Eksekusi Eksperimen Tuning Colab & Extraksi Metrik CSV](#sesi-5-eksekusi-eksperimen-tuning-colab--extraksi-metrik-csv)
6. [Sesi 6: Penyesuaian Pola Wildcard `.gitignore` untuk Checkpoint Rekursif (DBG-05)](#sesi-6-penyesuaian-pola-wildcard-gitignore-untuk-checkpoint-rekursif-dbg-05)
7. [Sesi 7: Pembaruan `docs/log_debugging.md` ke Versi 2.0](#sesi-7-pembaruan-docslog_debuggingmd-ke-versi-20)
8. [Sesi 8: Pembaruan & Finalisasi `README.md`](#sesi-8-pembaruan--finalisasi-readmemd)
9. [Sesi 9: Penyusunan Laporan Akhir Kelompok (`docs/laporan_akhir.md` Bab 1–5)](#sesi-9-penyusunan-laporan-akhir-kelompok-docslaporan_akhir-bab-15)

---

## Sesi 1: Inisiasi Log Interaksi & Formulasi Template Log LLM (`docs/log_interaksi.md`)
**Tanggal:** 07/11/2026

### 👤 Prompt 1.1 — Anggota 4 (Furqan):
> *"Saya Anggota 4 (Lead Documentarian & Error Handler) kelompok GEN-AI-AJA. Tugas utama saya adalah menyusun log interaksi LLM seluruh tim, mendokumentasikan debugging, dan menyusun laporan akhir proyek cGAN MNIST. Bagaimana struktur dokumen log interaksi yang transparan dan memenuhi standar rubrik 25% penilaian?"*

### 🤖 Respons LLM 1.1:
Halo Furqan! Struktur dokumen `log_interaksi.md` dirancang untuk merekam jejak berpikir (*chain of thought*) tim dari Sesi 0 hingga Sesi Final.

Struktur terstandarisasi mencakup:
1. **Identitas Dokumen & Tim:** Nama, NIM, peran 4 anggota, dan target modul.
2. **Daftar Isi Kronologi:** Navigasi berbasis anchor tautan per sesi.
3. **Kronologi Prompt & Respons:**
   - Text Prompt asli mahasiswa (tanpa rekayasa).
   - Analisis & Alasan Khusus pemilihan prompt.
   - Ringkasan output AI dan keputusan desain yang disepakati.
4. **Rekapitulasi Capaian Berkas:** Tabel matriks verifikasi ketersediaan dokumen di repositori.

---

## Sesi 2: Dokumentasi Debugging Awal & Isolasi Eror GUI Git (`docs/log_debugging.md` v1.0)
**Tanggal:** 07/11/2026

### 👤 Prompt 2.1 — Anggota 4 (Furqan):
> *"Anggota 1 dan 3 menemukan kendala saat commit via VS Code GUI dan warning CRLF di Windows. Tolong buatkan draf awal `docs/log_debugging.md` versi 1.0 yang mencatat ID masalah DBG-01, DBG-02, dan DBG-03."*

### 🤖 Respons LLM 2.1:
Draf `docs/log_debugging.md` v1.0 telah dibuat dengan tabel pencatatan eror terstruktur:
- **DBG-01 (LF to CRLF Warning):** Klasifikasi *Warning*. Konversi otomatis line-ending Git di Windows. Penanganan: Diabaikan karena tidak merusak kode.
- **DBG-02 (VS Code Commit Hang):** Klasifikasi *Tooling Error*. GUI VS Code tertahan karena kotak commit message kosong. Solusi: Eksekusi CLI `git commit -m "..."`.
- **DBG-03 (Pylance Torch Missing):** Klasifikasi *Environment Error*. Visual Studio Code tidak mengenali interpreter Conda/venv. Solusi: Re-select Python Interpreter ke lingkungan venv aktif.

---

## Sesi 3: Penanganan AppLocker & Pemblokiran DLL PyTorch Lokal (DBG-04)
**Tanggal:** 09/11/2026

### 👤 Prompt 3.1 — Anggota 4 (Furqan):
> *"Saat menjalankan `pip install -r requirements.txt` dan `python train.py` di terminal lokal, muncul eror berikut:*
> `Program 'pip.exe' failed to run: An Application Control policy has blocked this file`
> *dan saat memakai python -m pip install, muncul:*
> `ImportError: DLL load failed while importing _C: An Application Control policy has blocked this file`
> *Bagaimana analisis dan solusinya?"*

### 🤖 Respons LLM 3.1:
**Analisis Akar Masalah (DBG-04):**
Komputer lokal Anda dilindungi oleh kebijakan keamanan institusi (**Windows AppLocker / Application Control policy**). Kebijakan ini memblokir eksekusi berkas biner `.exe` dan pustaka C++ terkompilasi (`.pyd` / `.dll` milik `torch._C`) yang diunduh di luar direktori sistem Windows (`C:\Program Files`).

**Solusi & Rekomendasi:**
1. Penggunaan `python -m pip` hanya berhasil meloloskan proses pengunduhan paket, namun gagal saat memuat DLL dinamis PyTorch ke memori.
2. Rekomendasi utama: Beralih sepenuhnya ke **Google Colab** untuk mengeksekusi pelatihan cGAN. Colab menyediakan lingkungan Linux yang bebas blokir AppLocker serta dilengkapi akselerator GPU T4 gratis.

---
