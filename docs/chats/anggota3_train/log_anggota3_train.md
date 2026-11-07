# LOG INTERAKSI LLM — KELOMPOK GEN-AI-AJA

**Anggota 3 — Training Loop, Utilities & Hyperparameter Tuning Engineer**

- **Tanggal Sesi:** 07/11/2026
- **LLM yang digunakan:** Gemini / Claude (via Antigravity)
- **Target Modul:** `utils.py` dan `train.py` (Training Loop, Utilitas Visualisasi/Logging, & Hyperparameter Tuning)
- **Tautan Sesi Chat:** [ISI DENGAN SHARE LINK CHAT]

---

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI MODUL 1 — `utils.py`

- **Prompt Kelompok:**
  > "Peran Anda adalah Machine Learning Engineer spesialis PyTorch dan Generative Adversarial Networks (GAN). Saya bertugas sebagai Anggota 3 dalam proyek kelompok pembuatan Conditional GAN (cGAN) untuk dataset MNIST (from scratch). Tugas saya adalah menyusun utils.py, train.py, dan hyperparameter tuning. Pertama, buatkan skrip lengkap `utils.py` modular tanpa dependensi tutorial luar. File ini harus memiliki fungsi: 1. `set_seed(seed=42)`, 2. `save_checkpoint(state, filepath)`, 3. `log_metrics_to_csv(filepath, epoch, d_loss, g_loss)`, 4. `save_sample_images(generator, epoch, fixed_noise, fixed_labels, device, output_dir)`, 5. `plot_loss_curves(csv_path, output_image_path)`. Pastikan setiap fungsi memiliki type hints, docstring ringkas, dan error handling pembuatan direktori otomatis jika belum ada."

- **Alasan Khusus:**
  Memisahkan fungsi-fungsi I/O, reprodusibilitas, visualisasi sampel 10 kelas, dan plotting metrik ke dalam `utils.py` agar `train.py` tetap fokus pada logika pelatihan adversarial dan langsung mematuhi jalur direktori di `config.py`.

- **Output Model (Ringkasan):**
  | Fungsi | Kegunaan |
  | :--- | :--- |
  | `set_seed(seed=SEED)` | Mengunci seed `random`, NumPy, `PYTHONHASHSEED`, dan PyTorch (CPU & CUDA deterministik) |
  | `denormalize(tensor)` | Mengubah rentang piksel citra dari `[-1, 1]` (output `Tanh`) kembali ke `[0, 1]` |
  | `save_checkpoint(state, filepath)` & `save_checkpoints(...)` | Menyimpan bobot model secara atomik (`.tmp` → `os.replace`) ke `outputs/checkpoints/` dengan key `"generator_state_dict"` yang kompatibel dengan `generate.py` |
  | `log_metrics_to_csv(...)` | Mencatat `epoch, d_loss, g_loss, lr_g, lr_d, batch_size` ke `outputs/logs/`; membuat header otomatis bila file belum ada |
  | `save_sample_images(...)` & `save_sample_grid(...)` | Membangkitkan grid sampel digit 0–9 terurut (10 kolom) dalam mode `.eval()` ke `outputs/samples/epoch_XXX.png` |
  | `plot_loss_curves(...)` & `plot_experiment_comparison(...)` | Membaca CSV dengan Pandas dan menyimpan grafik kurva Loss D vs Loss G (tunggal maupun gabungan antar-eksperimen) ke `outputs/plots/` |

---

### 2. TAHAP INISIASI MODUL 2 — `train.py`

- **Prompt Kelompok:**
  > "Sekarang kita buat `train.py`. Arsitektur proyek menggunakan struktur file berikut: `config.py`, `models.py`, `data.py`, `utils.py`. Tolong susun `train.py` dengan kriteria: 1. Menggunakan loss function `torch.nn.BCEWithLogitsLoss`. 2. Training loop cGAN standar 2 langkah: Langkah 1: Update Discriminator menggunakan real data + fake data yang dikondisikan dengan label kelas. Langkah 2: Update Generator untuk memprediksi fake data dikondisikan label agar dinilai real oleh Discriminator. 3. Mendukung argumen CLI (via argparse) untuk fleksibilitas hyperparameter tuning: `--epochs`, `--lr`, `--batch_size`, `--label_smoothing` (boolean), dan `--exp_name`. 4. fixed_noise dan fixed_labels dibuat di awal training agar sampel gambar tiap epoch konsisten. 5. Memanggil logger CSV tiap epoch dan menyimpan checkpoint model terbaik/terakhir."

- **Alasan Khusus:**
  Menyusun *training loop* cGAN dua langkah yang terintegrasi penuh dengan `ConditionalGenerator` dan `ConditionalDiscriminator` dari Anggota 1 serta `get_dataloader` dari Anggota 2, sekaligus mendukung parameterisasi penuh melalui CLI (`argparse`) untuk eksperimen *hyperparameter tuning*.

- **Output Model (Ringkasan):**
  - Menggunakan `nn.BCEWithLogitsLoss()` dan `optim.Adam(..., betas=(BETA1, BETA2))`.
  - Memisahkan aliran gradien langkah D dengan `fake_images.detach()` dan mengevaluasi ulang `fake_images` pada langkah G terhadap target `1.0`.
  - Menyediakan argumen CLI: `--epochs`, `--batch_size`, `--lr`, `--lr_g`, `--lr_d`, `--label_smoothing`, `--exp_name`, dan `--compare_with`.

---

### 3. TAHAP DEBUGGING & HYPERPARAMETER TUNING (Mengatasi Discriminator Overpower & Mode Collapse)

- **Pesan Eror / Gejala Klinis Pelatihan:**
  > "Training baseline cGAN saya menghasilkan kurva loss berikut:
  > - Discriminator loss turun mendekati 0.05 pada epoch 10.
  > - Generator loss melonjak di atas 4.5 dan kualitas gambar output angka 0-9 mengalami blur serta beberapa kelas mengalami mode collapse (hanya menghasilkan bentuk angka 1 dan 8)."

- **Prompt Perbaikan ke LLM:**
  > "Tolong rekomendasikan strategi hyperparameter tuning PyTorch untuk mengatasi Discriminator yang terlalu kuat:
  > 1. Apakah penyesuaian learning rate (misal Two-Time-Scale Update Rule / TTUR) atau One-Sided Label Smoothing (misal real target = 0.9) lebih efektif?
  > 2. Berikan instruksi modifikasi kode pada `train.py` untuk mengaktifkan opsi tersebut dan bagaimana cara membandingkan kedua log CSV eksperimen menggunakan grafik plot gabungan."

- **Analisis Matematis & Solusi Kelompok:**
  1. **Penyebab Matematis (Vanishing Gradient & Mode Collapse):**
     Saat $\mathcal{L}_D$ terus turun (pada run `baseline` 30 epoch, $\mathcal{L}_D$ turun dari `0.3886` ke `0.2323` sementara $\mathcal{L}_G$ naik dari `1.8626` ke `3.1047`), Discriminator makin dominan memisahkan distribusi citra asli dan citra palsu. Karena kurva sigmoid mendatar (jenuh) pada logit ekstrem, gradien yang diterima Generator menyusut dan mendorong *partial mode collapse*.
  2. **Perbandingan Efektivitas One-Sided Label Smoothing vs TTUR:**
     - **One-Sided Label Smoothing (`real_target = 0.9`)** menahan logit optimal Discriminator pada $D^*(x) \le \log(9) \approx 2.197$, sehingga $\mathcal{L}_D$ memiliki *lower bound* teoritis $\approx 0.325$. Ini paling efektif untuk menahan kejatuhan $\mathcal{L}_D$ dan lonjakan $\mathcal{L}_G$.
     - **TTUR (`--lr_g 0.0002 --lr_d 0.0001`)** memperlambat laju adaptasi Discriminator sebesar $0.5\times$ agar Generator sempat mempelajari perbedaan fitur tiap kelas digit `0–9`.
     - **Strategi Terbaik:** Kombinasi keduanya (`--label_smoothing` + TTUR `--lr_g 0.0002 --lr_d 0.0001`).

---

## HASIL PELATIHAN BASELINE (30 EPOCH — `outputs/logs/metrics_baseline.csv`)

Konfigurasi: `epochs=30`, `batch_size=64`, `lr_g=0.0002`, `lr_d=0.0002`, `betas=(0.5, 0.999)`, `label_smoothing=False`.

| Epoch | Loss Discriminator (`d_loss`) | Loss Generator (`g_loss`) | LR_G | LR_D | Batch Size |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.388593 | 1.862647 | 0.0002 | 0.0002 | 64 |
| 5 | 0.366775 | 2.120289 | 0.0002 | 0.0002 | 64 |
| 10 | 0.323158 | 2.449079 | 0.0002 | 0.0002 | 64 |
| 15 | 0.303578 | 2.604213 | 0.0002 | 0.0002 | 64 |
| 20 | 0.266542 | 2.873219 | 0.0002 | 0.0002 | 64 |
| 25 | 0.274892 | 2.852629 | 0.0002 | 0.0002 | 64 |
| 30 | 0.232262 | 3.104701 | 0.0002 | 0.0002 | 64 |

Artefak yang dihasilkan:
- Checkpoint: `outputs/checkpoints/baseline/best.pth`, `outputs/checkpoints/baseline/last.pth`, serta `outputs/checkpoints/generator_latest.pth` & `discriminator_latest.pth`
- Log CSV: `outputs/logs/metrics_baseline.csv` & `outputs/logs/training_metrics.csv`
- Grafik Loss: `outputs/plots/loss_curves.png` & `outputs/plots/loss_curves_baseline.png`
- Sampel Grid 0–9: `outputs/samples/baseline/epoch_001.png` s.d. `epoch_030.png`
