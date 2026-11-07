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
  [tempel pesan eror utuh di sini, atau output sanity check jika berhasil]
  ```
- **Prompt Perbaikan ke LLM:** "[tulis prompt yang kamu kirim ke LLM untuk memperbaiki eror]"
- **Analisis & Solusi Kelompok:** [jelaskan penyebab eror dan perbaikannya]

---

## HASIL UJI (diisi setelah menjalankan)

```text
[tempel output `python data.py` di sini]
```

```text
[tempel output `python verify_integration.py` di sini — status data harus "AKTIF (DataLoader Anggota 2 terdeteksi)"]
```

Pratinjau data: `outputs/samples/real_mnist_preview.png`
