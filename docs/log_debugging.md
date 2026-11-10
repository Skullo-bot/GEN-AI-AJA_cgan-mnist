# LOG DEBUGGING — KELOMPOK GEN-AI-AJA

| Item | Keterangan |
| :--- | :--- |
| **Proyek** | Conditional GAN (cGAN) Klasik pada MNIST (PyTorch, *Vibe Coding from scratch*) |
| **Mata Kuliah** | Generative AI (CAK4KBB3) |
| **Disusun oleh** | Anggota 4 (Lead Documentarian & Error Handler) |
| **Versi** | v2.0 — 10/11/2026 (Diperbarui setelah eksperimen tuning selesai) |
| **Sumber** | Log mentah `docs/chats/`, `outputs/logs/metrics_*.csv`, `outputs/samples/`, dan repositori |

---

## 1. Ringkasan Seluruh Catatan

| ID | Modul / Area | Kategori | Gejala Singkat | Pelapor | Status |
| :-: | :--- | :--- | :--- | :-: | :-: |
| DBG-01 | Git (`.gitignore`) | Peringatan | `LF will be replaced by CRLF` | 1 | Selesai |
| DBG-02 | Git (VS Code GUI) | Eror Alat | *Commit* berputar terus (*hang*) | 1 | Selesai |
| DBG-03 | Lingkungan Python | Eror Impor | `Import "torch" could not be resolved` | 1 | Selesai |
| DBG-04 | `train.py` / Tuning | Dinamika Training | D dominan pada baseline; diatasi dengan Smooth & TTUR | 3, 4 | **Selesai** |
| DBG-05 | `.gitignore` | Konfigurasi | Checkpoint `.pth` ter-commit; disesuaikan wildcard | 4 | **Selesai** |
| PRV-01 | `data.py` | Pencegahan | Risiko batch ukuran 1 pada `BatchNorm1d` | 2 | Diterapkan |
| OBS-01 | `generate.py` | Verifikasi | Grid abu-abu pada kondisi *untrained* | 1 | Selesai |

---

## 2. Catatan Debugging Lingkungan & Git (DBG-01 s.d. DBG-03)

- **DBG-01:** Peringatan konversi konvensi akhir baris Windows (`CRLF`) ke Unix (`LF`). Tidak mengganggu proses *push* Git.
- **DBG-02:** *Commit* GUI VS Code tertahan karena pesan *commit* kosong. Diselesaikan via CLI terminal `git commit -m "..."`.
- **DBG-03:** Pylance tidak mengenali pustaka `torch`. Diselesaikan dengan menyelaraskan Python Interpreter ke lingkungan Conda/venv aktif.

---

## 3. DBG-04 — Evaluasi Performa Baseline vs Eksperimen Tuning (v2.0)

**Status: SELESAI** · **Modul:** `train.py`, `utils.py`

### Data Kuantitatif Akhir (Epoch 30)

| Eksperimen | Parameter (`lr_g` / `lr_d`) | Label Smoothing | `d_loss` (Ep 30) | `g_loss` (Ep 30) | Status Ekuilibrium |
| :--- | :-: | :-: | :-: | :-: | :--- |
| `baseline` | 0.0002 / 0.0002 | Nonaktif | 0.2323 | 3.1047 | D terlalu dominan, G loss naik |
| `smooth` | 0.0002 / 0.0002 | Aktif (0.9) | 0.3462 | 2.6766 | *Overconfidence* D tertahan |
| `ttur` | 0.0002 / 0.0001 | Nonaktif | 0.3783 | 1.9799 | G loss turun drastis (−36.2%) |
| `smooth_ttur` | 0.0002 / 0.0001 | Aktif (0.9) | **0.4427** | **1.9862** | **Paling Stabil ($pprox \ln 2$)** |

### Kesimpulan Solusi DBG-04
Penggunaan teknik **Two Time-Scale Update Rule (TTUR)** bersama **One-Sided Label Smoothing** berhasil menghentikan dominasi Discriminator, menekan lonjakan `g_loss` dari 3.1047 menjadi 1.9862, dan meningkatkan keragaman citra sintetis yang dihasilkan Generator.

---

## 4. DBG-05 — Penyesuaian Pola `.gitignore` untuk Berkas Checkpoint

**Status: SELESAI** · **Modul:** `.gitignore`

- **Masalah:** Pola awal `outputs/checkpoints/*.pth` tidak menjangkau subfolder per eksperimen (seperti `outputs/checkpoints/baseline/best.pth`), sehingga file bobot besar (±30 MB) terikut ke repositori.
- **Solusi:** Pola diubah menjadi `outputs/checkpoints/**/*.pth` dan `outputs/checkpoints/**/*.pt` agar seluruh bobot di subfolder diabaikan secara rekursif oleh Git.

---

## 5. Ringkasan Eksekusi Eksperimen Final

 Seluruh berkas metrik `metrics_smooth.csv`, `metrics_ttur.csv`, dan `metrics_smooth_ttur.csv` telah tersimpan di folder `outputs/logs/` dan siap dirujuk dalam laporan akhir.
