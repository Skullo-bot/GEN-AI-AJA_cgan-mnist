# LOG INTERAKSI LLM — KELOMPOK GEN-AI-AJA

**Anggota 2 — Data Pipeline & Preprocessing Engineer**

- **Tanggal Sesi:** 07/11/2026
- **LLM yang digunakan:** Claude (Anthropic)
- **Target Modul:** `data.py` (pipeline dataset MNIST, normalisasi, augmentasi, DataLoader)
- **Tautan Sesi Chat:** [ISI DENGAN SHARE LINK CHAT]

---

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Kode Awal)

- **Prompt Kelompok:**
  > "Saya adalah anggota dua (2), coba tolong kerjakan apa yang seharusnya jadi tugas saya, bisa dilihat di GitHub tersebut untuk file pengerjaan." (disertai panduan proyek dan tautan repositori kelompok)

- **Alasan Khusus:**
  LLM diberi akses ke repositori agar membaca langsung kontrak dari Anggota 1 (`config.py`, `models.py`, `verify_integration.py`, dan system prompt Anggota 2 di `docs/system_prompt.md`). Dengan begitu `data.py` tidak memakai angka hardcode dan shape-nya pasti cocok dengan Generator/Discriminator.

- **Output Model (ringkasan):**
  | Fungsi | Kegunaan |
  | :--- | :--- |
  | `get_mnist_transforms(augment=False)` | `Resize(28)` → `ToTensor()` [0, 1] → `Normalize(0.5, 0.5)` [-1, 1]; opsi augmentasi ringan `RandomAffine` (rotasi ±10°, geser ±2 px) |
  | `get_mnist_dataset(train, augment)` | Mengunduh/memuat `torchvision.datasets.MNIST` ke `DATA_DIR` |
  | `get_dataloader(batch_size, train, augment, num_workers)` | DataLoader dengan `torch.Generator` ber-seed 42 (reprodusibel), `shuffle` & `drop_last` hanya untuk data latih |
  | `get_class_distribution(dataset)` | Menghitung jumlah sampel per digit 0–9 untuk analisis data |
  | `validate_batch(images, labels)` | Assert kontrak: `images` float32 `[B, 1, 28, 28]` di [-1, 1], `labels` int64 `[B]` di 0–9 |
  | `save_real_preview(...)` | Menyimpan grid citra asli ke `outputs/samples/real_mnist_preview.png` |

  Cuplikan krusial:
  ```python
  transforms.ToTensor(),                      # [1, 28, 28], float32, [0, 1]
  transforms.Normalize(NORM_MEAN, NORM_STD),  # (x - 0.5) / 0.5 -> [-1, 1]
  ...
  loader_generator = torch.Generator()
  loader_generator.manual_seed(SEED)          # urutan shuffle selalu sama
  ```

- **Keputusan desain yang dijelaskan LLM:**
  1. Normalisasi ke [-1, 1] karena Generator memakai aktivasi `Tanh`; citra asli dan palsu harus berada pada rentang yang sama di mata Discriminator.
  2. `drop_last=True` pada data latih agar tidak ada batch berukuran 1 yang membuat `BatchNorm1d` di Generator eror.
  3. Augmentasi dimatikan secara default karena GAN harus mempelajari distribusi asli MNIST; augmentasi berat bisa ikut ditiru Generator.
  4. `NUM_WORKERS=0` (dari config) agar aman di Windows.

### 2. TAHAP PENANGANAN EROR / DEBUGGING

> Isi bagian ini setelah menjalankan `python data.py` dan `python verify_integration.py` di laptop sendiri. Jika tidak ada eror, tulis "Tidak ditemukan eror" dan tempel output terminal sebagai bukti.

- **Pesan Eror dari Notebook/Terminal:**
  ```text
  [MNIST akan terunduh otomatis ke folder dataset/.]
  ```
- **Prompt Perbaikan ke LLM:** "[tulis prompt yang kamu kirim ke LLM untuk memperbaiki eror]"
- **Analisis & Solusi Kelompok:** [jelaskan penyebab eror dan perbaikannya]

---

## HASIL UJI (diisi setelah menjalankan)

```text
[
  [100% 9.91M/9.91M [00:00<00:00, 18.3MB/s]
  100% 28.9k/28.9k [00:00<00:00, 484kB/s]
  100% 1.65M/1.65M [00:00<00:00, 4.63MB/s]
  100% 4.54k/4.54k [00:00<00:00, 18.9MB/s]
  === SANITY CHECK: data.py ===
  Jumlah data latih      : 60,000 citra (937 batch)
  Jumlah data uji        : 10,000 citra (157 batch)
  Shape batch images     : [64, 1, 28, 28]  (Target: [64, 1, 28, 28])
  Shape batch labels     : [64]  (Target: [64])
  Tipe data images/label : torch.float32 / torch.int64
  Rentang piksel         : [-1.000, 1.000]  (Target: [-1, 1])
  Mean / Std batch       : -0.742 / 0.614
  Label pada batch ini   : [8, 1, 4, 6, 3, 4, 0, 2, 7, 6, 4, 0, 3, 1, 5, 6] ...
  Distribusi kelas (data latih):
  Digit 0:  5923 ( 9.87%)
  Digit 1:  6742 (11.24%)
  Digit 2:  5958 ( 9.93%)
  Digit 3:  6131 (10.22%)
  Digit 4:  5842 ( 9.74%)
  Digit 5:  5421 ( 9.04%)
  Digit 6:  5918 ( 9.86%)
  Digit 7:  6265 (10.44%)
  Digit 8:  5851 ( 9.75%)
  Digit 9:  5949 ( 9.91%)
  Reprodusibel (seed 42) : True
  Batch augmentasi       : [8, 1, 28, 28] -> VALID
  Pratinjau disimpan     : /content/GEN-AI-AJA_cgan-mnist/outputs/samples/real_mnist_preview.png
  Status                 : Data pipeline MNIST VALID & siap dipakai train.py!]
]
```

```text
[
  === VERIFIKASI INTEGRASI MODUL cGAN MNIST ===
  Device Eksekusi       : cuda
  Status Modul data.py  : AKTIF (DataLoader Anggota 2 terdeteksi)
  Shape Citra Real/Fake : [8, 1, 28, 28] / [8, 1, 28, 28]
  Rentang Piksel Fake   : [-0.682, 0.649]
  Shape Output Logit D  : [8, 1]
  Uji Hitung Loss D & G : Loss_D = 0.9063 | Loss_G = 0.8154
  Status Integrasi      : SELURUH KONTRAK SHAPE & BACKPROPAGATION VALID!
]
```

Pratinjau data: `outputs/samples/real_mnist_preview.png`
