from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    DATA_DIR,
    IMG_CHANNELS,
    IMG_SIZE,
    NORM_MEAN,
    NORM_STD,
    NUM_CLASSES,
    NUM_WORKERS,
    SAMPLE_DIR,
    SEED,
    ensure_directories,
)


# ---------------------------------------------------------------------------
# 1. Transformasi / Preprocessing
# ---------------------------------------------------------------------------
def get_mnist_transforms(augment: bool = False) -> transforms.Compose:
    """
    Menyusun pipeline transformasi citra MNIST.

    Args:
        augment: Jika True, menambahkan augmentasi ringan (rotasi kecil ±10°
                 dan pergeseran ±2 piksel). Default False karena GAN harus
                 belajar distribusi asli MNIST; augmentasi berat dapat membuat
                 Generator meniru artefak augmentasi.

    Returns:
        transforms.Compose yang mengubah PIL Image -> Tensor [1, 28, 28] di [-1, 1].
    """
    pipeline = [transforms.Resize(IMG_SIZE)]  # PIL [28, 28] (aman bila IMG_SIZE diubah)

    if augment:
        pipeline.append(
            transforms.RandomAffine(
                degrees=10,                # rotasi kecil agar bentuk digit tetap terbaca
                translate=(2 / IMG_SIZE, 2 / IMG_SIZE),  # geser maksimal ±2 piksel
            )
        )

    pipeline += [
        transforms.ToTensor(),                     # [1, 28, 28], float32, [0, 1]
        transforms.Normalize(NORM_MEAN, NORM_STD),  # (x - 0.5) / 0.5 -> [-1, 1]
    ]
    return transforms.Compose(pipeline)


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------
def get_mnist_dataset(train: bool = True, augment: bool = False) -> Dataset:
    """
    Mengunduh (jika belum ada) dan memuat dataset MNIST ke folder DATA_DIR.

    Args:
        train  : True -> 60.000 data latih, False -> 10.000 data uji.
        augment: Meneruskan opsi augmentasi ke get_mnist_transforms().

    Returns:
        torchvision.datasets.MNIST; setiap item = (image [1, 28, 28], label int).
    """
    ensure_directories()  # memastikan folder dataset/ tersedia
    return datasets.MNIST(
        root=str(DATA_DIR),
        train=train,
        download=True,
        transform=get_mnist_transforms(augment=augment),
    )


# ---------------------------------------------------------------------------
# 3. DataLoader (fungsi utama yang dipakai train.py & verify_integration.py)
# ---------------------------------------------------------------------------
def get_dataloader(
    batch_size: int = BATCH_SIZE,
    train: bool = True,
    augment: bool = False,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    """
    Membuat DataLoader MNIST yang reprodusibel.

    Args:
        batch_size : Jumlah citra per batch (default BATCH_SIZE dari config).
        train      : True untuk data latih (di-shuffle), False untuk data uji.
        augment    : Aktifkan augmentasi ringan (default False).
        num_workers: Jumlah worker proses pemuatan data (default 0, aman di Windows).

    Returns:
        DataLoader yang setiap iterasinya menghasilkan tuple (images, labels):
            images: torch.float32 [B, 1, 28, 28], rentang [-1, 1]
            labels: torch.int64   [B], nilai 0..9
    """
    dataset = get_mnist_dataset(train=train, augment=augment)

    # Generator dengan seed tetap -> urutan shuffle sama setiap kali dijalankan
    loader_generator = torch.Generator()
    loader_generator.manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,                       # acak hanya untuk data latih
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # transfer ke GPU lebih cepat
        drop_last=train,                     # buang batch sisa agar BatchNorm stabil
        generator=loader_generator,
    )


# ---------------------------------------------------------------------------
# 4. Utilitas analisis data (untuk laporan & sanity check)
# ---------------------------------------------------------------------------
def get_class_distribution(dataset: Dataset) -> Dict[int, int]:
    """
    Menghitung jumlah sampel per kelas tanpa memuat gambar (cepat).

    Returns:
        Dict {digit: jumlah_sampel} untuk digit 0..NUM_CLASSES-1.
    """
    targets = torch.as_tensor(dataset.targets)               # [N]
    counts = torch.bincount(targets, minlength=NUM_CLASSES)  # [NUM_CLASSES]
    return {digit: int(counts[digit]) for digit in range(NUM_CLASSES)}


def validate_batch(images: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    """
    Memastikan satu batch sesuai kontrak shape & tipe data tim.

    Returns:
        (nilai_min, nilai_maks) piksel batch.
    """
    batch_size = images.size(0)
    assert images.shape == (batch_size, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), (
        f"Shape images salah: {tuple(images.shape)}, "
        f"target [B, {IMG_CHANNELS}, {IMG_SIZE}, {IMG_SIZE}]"
    )
    assert images.dtype == torch.float32, f"dtype images salah: {images.dtype}"
    assert labels.shape == (batch_size,), f"Shape labels salah: {tuple(labels.shape)}"
    assert labels.dtype == torch.int64, f"dtype labels salah: {labels.dtype}"

    img_min, img_max = images.min().item(), images.max().item()
    assert img_min >= -1.0 - 1e-6 and img_max <= 1.0 + 1e-6, (
        f"Rentang piksel di luar [-1, 1]: [{img_min:.3f}, {img_max:.3f}]"
    )
    assert labels.min().item() >= 0 and labels.max().item() < NUM_CLASSES, "Label di luar 0..9"
    return img_min, img_max


def save_real_preview(images: torch.Tensor, labels: torch.Tensor, filename: str = "real_mnist_preview.png") -> None:
    """
    Menyimpan pratinjau citra MNIST asli (maks 32 citra) ke outputs/samples/
    sebagai bukti visual preprocessing untuk laporan.
    """
    import matplotlib

    matplotlib.use("Agg")  # tanpa jendela GUI
    import matplotlib.pyplot as plt

    n_show = min(32, images.size(0))
    cols = 8
    rows = (n_show + cols - 1) // cols
    imgs = (images[:n_show].cpu() + 1.0) / 2.0  # [-1, 1] -> [0, 1] untuk ditampilkan

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx < n_show:
            ax.imshow(imgs[idx, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)  # [28, 28]
            ax.set_title(str(int(labels[idx])), fontsize=9)
    fig.suptitle("Sampel MNIST setelah preprocessing (label di atas citra)")
    fig.tight_layout()

    save_path = SAMPLE_DIR / filename
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Pratinjau disimpan     : {save_path}")


# ---------------------------------------------------------------------------
# 5. Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(SEED)

    train_loader = get_dataloader(batch_size=BATCH_SIZE, train=True)
    test_loader = get_dataloader(batch_size=BATCH_SIZE, train=False)

    images, labels = next(iter(train_loader))  # [B, 1, 28, 28], [B]
    img_min, img_max = validate_batch(images, labels)

    print("=== SANITY CHECK: data.py ===")
    print(f"Jumlah data latih      : {len(train_loader.dataset):,} citra ({len(train_loader)} batch)")
    print(f"Jumlah data uji        : {len(test_loader.dataset):,} citra ({len(test_loader)} batch)")
    print(f"Shape batch images     : {list(images.shape)}  (Target: [{BATCH_SIZE}, 1, 28, 28])")
    print(f"Shape batch labels     : {list(labels.shape)}  (Target: [{BATCH_SIZE}])")
    print(f"Tipe data images/label : {images.dtype} / {labels.dtype}")
    print(f"Rentang piksel         : [{img_min:.3f}, {img_max:.3f}]  (Target: [-1, 1])")
    print(f"Mean / Std batch       : {images.mean().item():.3f} / {images.std().item():.3f}")
    print(f"Label pada batch ini   : {labels[:16].tolist()} ...")

    print("Distribusi kelas (data latih):")
    distribution = get_class_distribution(train_loader.dataset)
    total = sum(distribution.values())
    for digit, count in distribution.items():
        print(f"  Digit {digit}: {count:>5} ({100 * count / total:5.2f}%)")

    # Cek reprodusibilitas: dua loader dengan seed sama -> batch pertama identik
    labels_again = next(iter(get_dataloader(batch_size=BATCH_SIZE, train=True)))[1]
    print(f"Reprodusibel (seed {SEED}) : {torch.equal(labels, labels_again)}")

    # Cek augmentasi tetap menjaga kontrak shape & rentang
    aug_images, aug_labels = next(iter(get_dataloader(batch_size=8, train=True, augment=True)))
    validate_batch(aug_images, aug_labels)
    print(f"Batch augmentasi       : {list(aug_images.shape)} -> VALID")

    try:
        save_real_preview(images, labels)
    except ImportError:
        print("matplotlib belum terpasang, pratinjau dilewati.")

    print("Status                 : Data pipeline MNIST VALID & siap dipakai train.py!")
