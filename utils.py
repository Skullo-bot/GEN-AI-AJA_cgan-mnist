"""
utils.py - Modul utilitas untuk pelatihan Conditional GAN (cGAN) MNIST.

Menyediakan fungsi reprodusibilitas, denormalisasi citra, penyimpanan sampel grid
angka 0-9, pencatatan metrik CSV, visualisasi kurva loss, dan manajemen checkpoint
sesuai kontrak integrasi tim.
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # Backend non-GUI agar aman dijalankan tanpa display window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torchvision.utils import save_image  # noqa: E402

from config import (  # noqa: E402
    BATCH_SIZE,
    BEST_DISC_PATH,
    BEST_GEN_PATH,
    CHECKPOINT_DIR,
    DEVICE,
    LATENT_DIM,
    LOG_DIR,
    LR_D,
    LR_G,
    METRICS_CSV_PATH,
    NUM_CLASSES,
    PLOT_DIR,
    SAMPLE_DIR,
    SEED,
    ensure_directories,
)

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Helper Internal: Resolusi Jalur & Direktori Otomatis
# ---------------------------------------------------------------------------
def _ensure_dir(directory: PathLike) -> Path:
    """Memastikan direktori target beserta parent-nya tersedia (dibuat otomatis jika belum ada)."""
    dir_path = Path(directory)
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Gagal membuat direktori '{dir_path}': {exc}") from exc
    return dir_path


def _resolve_filepath(filepath: PathLike, default_dir: Path) -> Path:
    """
    Menyelesaikan jalur file: jika `filepath` hanya berupa nama file (tanpa folder),
    maka diarahkan ke `default_dir`. Direktori parent dipastikan selalu tersedia.
    """
    path = Path(filepath)
    if not path.is_absolute() and path.parent == Path("."):
        path = default_dir / path
    _ensure_dir(path.parent)
    return path


# ---------------------------------------------------------------------------
# 1. Reprodusibilitas (Seed)
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    """
    Mengunci random seed untuk Python built-in random, NumPy, dan PyTorch (CPU & CUDA).

    Args:
        seed: Nilai integer seed reprodusibilitas (default mengikuti config.SEED = 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 2. Denormalisasi & Pembuatan Input Tetap (Fixed Noise & Labels)
# ---------------------------------------------------------------------------
def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Mengembalikan rentang piksel citra dari [-1, 1] (keluaran Tanh) ke [0, 1] untuk visualisasi.

    Args:
        tensor: Tensor citra dengan shape [B, C, H, W] atau [C, H, W] pada rentang [-1, 1].

    Returns:
        Tensor citra pada rentang [0, 1] dengan shape yang sama.
    """
    # Transformasi balik dari Normalize(mean=0.5, std=0.5): x_vis = (x * 0.5) + 0.5
    return (tensor.clamp(-1.0, 1.0) + 1.0) / 2.0


def make_fixed_inputs(
    n_per_class: int = 8,
    latent_dim: int = LATENT_DIM,
    num_classes: int = NUM_CLASSES,
    device: torch.device = DEVICE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Membuat pasangan fixed_noise dan fixed_labels terurut (0..9 berulang) untuk evaluasi visual.

    Returns:
        Tuple (fixed_noise, fixed_labels):
            - fixed_noise  : Tensor float32 [n_per_class * num_classes, latent_dim]
            - fixed_labels : Tensor int64   [n_per_class * num_classes]
    """
    total_samples = n_per_class * num_classes
    fixed_noise = torch.randn(total_samples, latent_dim, device=device)  # [N, 100]
    fixed_labels = torch.arange(num_classes, device=device, dtype=torch.long).repeat(
        n_per_class
    )  # [N] -> [0,1,...,9, 0,1,...,9, ...]
    return fixed_noise, fixed_labels


# ---------------------------------------------------------------------------
# 3. Penyimpanan Checkpoint Model
# ---------------------------------------------------------------------------
def save_checkpoint(state: Dict[str, Any], filepath: PathLike) -> Path:
    """
    Menyimpan dictionary checkpoint bobot model ke folder outputs/checkpoints/ secara atomik.

    Args:
        state: Dictionary berisi state_dict model, optimizer, atau metadata epoch.
        filepath: Nama file atau jalur tujuan penyimpanan (.pth).

    Returns:
        Path lengkap file checkpoint yang berhasil disimpan.
    """
    if not isinstance(state, dict):
        raise TypeError(f"Argumen `state` harus bertipe dict, diterima: {type(state).__name__}")

    target_path = _resolve_filepath(filepath, CHECKPOINT_DIR)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, target_path)
    except (OSError, RuntimeError) as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Gagal menyimpan checkpoint ke '{target_path}': {exc}") from exc
    return target_path


def save_checkpoints(
    generator: nn.Module,
    discriminator: nn.Module,
    epoch: int,
    d_loss: float = 0.0,
    g_loss: float = 0.0,
    gen_path: PathLike = BEST_GEN_PATH,
    disc_path: PathLike = BEST_DISC_PATH,
) -> Tuple[Path, Path]:
    """
    Menyimpan state_dict Generator dan Discriminator ke outputs/checkpoints/
    (kompatibel dengan `generate.py` yang memuat `weights_only=True`).

    Args:
        generator: Model ConditionalGenerator.
        discriminator: Model ConditionalDiscriminator.
        epoch: Nomor epoch saat checkpoint disimpan.
        d_loss: Nilai loss Discriminator terakhir.
        g_loss: Nilai loss Generator terakhir.
        gen_path: Jalur simpan checkpoint Generator (default BEST_GEN_PATH).
        disc_path: Jalur simpan checkpoint Discriminator (default BEST_DISC_PATH).

    Returns:
        Tuple (saved_gen_path, saved_disc_path).
    """
    gen_state = {
        "epoch": int(epoch),
        "generator_state_dict": generator.state_dict(),
        "g_loss": float(g_loss),
    }
    disc_state = {
        "epoch": int(epoch),
        "discriminator_state_dict": discriminator.state_dict(),
        "d_loss": float(d_loss),
    }
    saved_gen = save_checkpoint(gen_state, gen_path)
    saved_disc = save_checkpoint(disc_state, disc_path)
    return saved_gen, saved_disc


# ---------------------------------------------------------------------------
# 4. Pencatatan Log Metrik ke CSV
# ---------------------------------------------------------------------------
def log_metrics_to_csv(
    filepath: PathLike,
    epoch: int,
    d_loss: float,
    g_loss: float,
    lr_g: Optional[float] = None,
    lr_d: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> Path:
    """
    Mencatat metrik pelatihan per epoch ke file CSV di outputs/logs/.
    Jika file belum ada atau masih kosong, header dibuat otomatis terlebih dahulu.

    Args:
        filepath: Nama file atau jalur CSV tujuan (mis. METRICS_CSV_PATH).
        epoch: Nomor epoch saat ini.
        d_loss: Nilai rata-rata loss Discriminator pada epoch tersebut.
        g_loss: Nilai rata-rata loss Generator pada epoch tersebut.
        lr_g: (Opsional) Learning rate Generator untuk pencatatan hyperparameter tuning.
        lr_d: (Opsional) Learning rate Discriminator.
        batch_size: (Opsional) Ukuran batch.

    Returns:
        Path lengkap file CSV yang diperbarui.
    """
    csv_path = _resolve_filepath(filepath, LOG_DIR)

    d_val = float(d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss)
    g_val = float(g_loss.item() if isinstance(g_loss, torch.Tensor) else g_loss)

    include_hparams = (lr_g is not None) or (lr_d is not None) or (batch_size is not None)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, mode="r", encoding="utf-8") as existing_file:
            first_line = existing_file.readline().strip()
            include_hparams = "lr_g" in first_line

    write_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)

    try:
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if include_hparams:
                if write_header:
                    writer.writerow(["epoch", "d_loss", "g_loss", "lr_g", "lr_d", "batch_size"])
                writer.writerow([
                    int(epoch),
                    f"{d_val:.6f}",
                    f"{g_val:.6f}",
                    f"{float(lr_g if lr_g is not None else LR_G):.6g}",
                    f"{float(lr_d if lr_d is not None else LR_D):.6g}",
                    int(batch_size if batch_size is not None else BATCH_SIZE),
                ])
            else:
                if write_header:
                    writer.writerow(["epoch", "d_loss", "g_loss"])
                writer.writerow([int(epoch), f"{d_val:.6f}", f"{g_val:.6f}"])
    except OSError as exc:
        raise OSError(f"Gagal menulis log metrik ke '{csv_path}': {exc}") from exc

    return csv_path


# ---------------------------------------------------------------------------
# 5. Visualisasi Sampel Grid Angka 0-9
# ---------------------------------------------------------------------------
@torch.no_grad()
def save_sample_images(
    generator: nn.Module,
    epoch: int,
    fixed_noise: torch.Tensor,
    fixed_labels: torch.Tensor,
    device: torch.device = DEVICE,
    output_dir: PathLike = SAMPLE_DIR,
) -> Path:
    """
    Mengenerate grid visualisasi angka 0 sampai 9 (10 kelas) dari fixed_noise dan
    fixed_labels, lalu menyimpannya sebagai file PNG di outputs/samples/.

    Args:
        generator: Model ConditionalGenerator.
        epoch: Nomor epoch saat ini (digunakan untuk penamaan file).
        fixed_noise: Tensor latent noise dengan shape [N, LATENT_DIM].
        fixed_labels: Tensor label kelas dengan shape [N] (0..9).
        device: Compute device (CPU/CUDA).
        output_dir: Direktori penyimpanan file PNG (default outputs/samples/).

    Returns:
        Path lengkap file PNG sampel yang disimpan.
    """
    if fixed_noise.size(0) != fixed_labels.size(0):
        raise ValueError(
            f"Ukuran batch fixed_noise ({fixed_noise.size(0)}) != fixed_labels ({fixed_labels.size(0)})."
        )

    target_dir = _ensure_dir(output_dir)
    save_path = target_dir / f"epoch_{int(epoch):03d}.png"

    was_training = generator.training
    generator.eval()  # Gunakan running stats BatchNorm selama inferensi sampel
    try:
        z = fixed_noise.to(device)                # [N, LATENT_DIM]
        labels = fixed_labels.to(device).long()   # [N]
        fake_images = generator(z, labels)        # [N, 1, 28, 28] pada rentang [-1, 1]
        vis_images = denormalize(fake_images)     # [N, 1, 28, 28] pada rentang [0, 1]

        save_image(
            vis_images.cpu(),
            save_path,
            nrow=NUM_CLASSES,
            padding=2,
            pad_value=1.0,
        )
    except OSError as exc:
        raise OSError(f"Gagal menyimpan citra sampel ke '{save_path}': {exc}") from exc
    finally:
        generator.train(was_training)  # Kembalikan mode training semula

    return save_path


@torch.no_grad()
def save_sample_grid(
    generator: nn.Module,
    epoch: int,
    device: torch.device = DEVICE,
    save_dir: PathLike = SAMPLE_DIR,
    n_rows: int = 5,
) -> Path:
    """
    Helper alternatif sesuai kontrak spesifikasi: membangkitkan grid terurut angka 0-9
    (tanpa perlu mengoper fixed_noise dari luar) dan menyimpannya sebagai PNG.
    """
    fixed_noise, fixed_labels = make_fixed_inputs(n_per_class=n_rows, device=device)
    return save_sample_images(
        generator=generator,
        epoch=epoch,
        fixed_noise=fixed_noise,
        fixed_labels=fixed_labels,
        device=device,
        output_dir=save_dir,
    )


# ---------------------------------------------------------------------------
# 6. Visualisasi Kurva Loss D vs Loss G
# ---------------------------------------------------------------------------
def plot_loss_curves(
    csv_path: PathLike = METRICS_CSV_PATH,
    output_image_path: PathLike = PLOT_DIR / "loss_curves.png",
) -> Path:
    """
    Membaca log CSV menggunakan Pandas dan menyimpan grafik perbandingan Loss D vs Loss G
    menggunakan Matplotlib ke outputs/plots/.

    Args:
        csv_path: Jalur file CSV log metrik (di outputs/logs/).
        output_image_path: Jalur file PNG output (atau direktori tujuan di outputs/plots/).

    Returns:
        Path lengkap file PNG grafik yang disimpan.
    """
    resolved_csv = _resolve_filepath(csv_path, LOG_DIR)
    if not resolved_csv.exists():
        raise FileNotFoundError(f"File CSV log metrik tidak ditemukan: '{resolved_csv}'")

    df = pd.read_csv(resolved_csv)
    required_cols = {"epoch", "d_loss", "g_loss"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak ditemukan di CSV: {sorted(missing_cols)}")
    if df.empty:
        raise ValueError(f"File CSV '{resolved_csv}' kosong, belum ada data untuk di-plot.")

    out_candidate = Path(output_image_path)
    if out_candidate.exists() and out_candidate.is_dir():
        save_path = out_candidate / f"{resolved_csv.stem}_loss_curves.png"
    elif out_candidate.suffix == "":
        save_path = _ensure_dir(out_candidate) / f"{resolved_csv.stem}_loss_curves.png"
    else:
        save_path = _resolve_filepath(out_candidate, PLOT_DIR)

    marker = "o" if len(df) <= 50 else None
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        ax.plot(
            df["epoch"],
            df["d_loss"],
            label="Discriminator Loss (D)",
            color="tab:red",
            marker=marker,
            markersize=4,
            linewidth=1.8,
        )
        ax.plot(
            df["epoch"],
            df["g_loss"],
            label="Generator Loss (G)",
            color="tab:blue",
            marker=marker,
            markersize=4,
            linewidth=1.8,
        )
        ax.set_title("Kurva Pelatihan Conditional GAN MNIST (Loss D vs Loss G)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCEWithLogitsLoss")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    except OSError as exc:
        raise OSError(f"Gagal menyimpan plot kurva loss ke '{save_path}': {exc}") from exc
    finally:
        plt.close(fig)

    return save_path


def plot_experiment_comparison(
    csv_paths: Dict[str, PathLike],
    output_image_path: PathLike = PLOT_DIR / "comparison_loss_curves.png",
) -> Path:
    """
    Membaca beberapa file CSV eksperimen (mis. baseline vs label_smoothing vs TTUR)
    dan memplot grafik perbandingan gabungan Loss D dan Loss G berdampingan.

    Args:
        csv_paths: Dictionary {nama_eksperimen: jalur_csv}.
        output_image_path: Jalur simpan PNG grafik perbandingan di outputs/plots/.

    Returns:
        Path lengkap file PNG grafik gabungan yang disimpan.
    """
    if not csv_paths:
        raise ValueError("`csv_paths` tidak boleh kosong.")

    save_path = _resolve_filepath(output_image_path, PLOT_DIR)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for exp_label, path_like in csv_paths.items():
        resolved_csv = _resolve_filepath(path_like, LOG_DIR)
        if not resolved_csv.exists():
            raise FileNotFoundError(f"File CSV untuk '{exp_label}' tidak ditemukan: '{resolved_csv}'")
        df = pd.read_csv(resolved_csv)
        axes[0].plot(df["epoch"], df["d_loss"], label=f"{exp_label} (D)", linewidth=1.8)
        axes[1].plot(df["epoch"], df["g_loss"], label=f"{exp_label} (G)", linewidth=1.8)

    axes[0].set_title("Perbandingan Discriminator Loss (D)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCEWithLogitsLoss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="best")

    axes[1].set_title("Perbandingan Generator Loss (G)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BCEWithLogitsLoss")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="best")

    fig.suptitle("Evaluasi Hyperparameter Tuning cGAN MNIST: Baseline vs Optimasi", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path



# ---------------------------------------------------------------------------
# Sanity Check Mandiri: python utils.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from models import ConditionalDiscriminator, ConditionalGenerator, initialize_weights

    set_seed(SEED)
    ensure_directories()

    netG = ConditionalGenerator().to(DEVICE)
    netD = ConditionalDiscriminator().to(DEVICE)
    initialize_weights(netG)
    initialize_weights(netD)

    fixed_z, fixed_y = make_fixed_inputs(n_per_class=4, device=DEVICE)
    sample_file = save_sample_images(netG, epoch=0, fixed_noise=fixed_z, fixed_labels=fixed_y, device=DEVICE)

    test_csv_name = "sanity_check_metrics.csv"
    for ep in range(1, 4):
        csv_file = log_metrics_to_csv(test_csv_name, epoch=ep, d_loss=0.65 / ep, g_loss=1.10 * ep)
    plot_file = plot_loss_curves(csv_file, "sanity_check_loss.png")

    ckpt_file = save_checkpoint({"epoch": 3, "generator_state_dict": netG.state_dict()}, "sanity_check_gen.pth")

    print("=== SANITY CHECK: utils.py ===")
    print(f"Device Aktif     : {DEVICE} (Seed: {SEED})")
    print(f"Sampel Grid PNG  : {sample_file}")
    print(f"File Log CSV     : {csv_file}")
    print(f"Plot Loss PNG    : {plot_file}")
    print(f"Checkpoint Model : {ckpt_file}")

    for temp_file in (sample_file, csv_file, plot_file, ckpt_file):
        Path(temp_file).unlink(missing_ok=True)
    print("Status           : Seluruh fungsi utils.py terverifikasi & artefak uji dibersihkan!")

