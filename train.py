"""
train.py - Skrip utama pelatihan Conditional GAN (cGAN) MNIST & Hyperparameter Tuning.

Mengintegrasikan:
    - config.py : Konstanta arsitektur, hyperparameter default, dan jalur direktori.
    - data.py   : Pipeline DataLoader MNIST dengan normalisasi [-1, 1].
    - models.py : Arsitektur ConditionalGenerator, ConditionalDiscriminator, & initialize_weights.
    - utils.py  : Reprodusibilitas, logging CSV, visualisasi sampel 0-9, plot loss, & checkpoint.

Contoh eksekusi CLI (Hyperparameter Tuning):
    python train.py --exp_name baseline
    python train.py --lr_g 0.0002 --lr_d 0.0001 --batch_size 64 --epochs 30 --exp_name ttur_lr
    python train.py --lr 0.0002 --batch_size 128 --label_smoothing --exp_name bs128_smooth
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    BATCH_SIZE,
    BEST_DISC_PATH,
    BEST_GEN_PATH,
    BETA1,
    BETA2,
    CHECKPOINT_DIR,
    DEVICE,
    EPOCHS,
    LATENT_DIM,
    LOG_DIR,
    LR_D,
    LR_G,
    METRICS_CSV_PATH,
    NUM_CLASSES,
    PLOT_DIR,
    SAMPLE_DIR,
    SAMPLE_INTERVAL,
    SEED,
    ensure_directories,
)
from data import get_dataloader
from models import ConditionalDiscriminator, ConditionalGenerator, initialize_weights
from utils import (
    log_metrics_to_csv,
    make_fixed_inputs,
    plot_experiment_comparison,
    plot_loss_curves,
    save_checkpoint,
    save_checkpoints,
    save_sample_images,
    set_seed,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

REAL_LABEL_SMOOTH: float = 0.9  # One-sided label smoothing untuk mencegah Discriminator overconfident


# ---------------------------------------------------------------------------
# 1. Konfigurasi Argumen CLI (Hyperparameter Tuning)
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Memproses argumen baris perintah (CLI) untuk eksperimen hyperparameter tuning."""
    parser = argparse.ArgumentParser(
        description="Pelatihan Conditional GAN (cGAN) MNIST & Hyperparameter Tuning"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Jumlah epoch pelatihan (default: {EPOCHS})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Ukuran batch DataLoader (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr_g",
        type=float,
        default=LR_G,
        help=f"Learning rate Generator (default: {LR_G})",
    )
    parser.add_argument(
        "--lr_d",
        type=float,
        default=LR_D,
        help=f"Learning rate Discriminator (default: {LR_D})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Opsional: jika diisi, menetapkan nilai learning rate yang sama untuk G dan D",
    )
    parser.add_argument(
        "--label_smoothing",
        action="store_true",
        help=f"Aktifkan one-sided label smoothing (target real D = {REAL_LABEL_SMOOTH})",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="baseline",
        help="Nama eksperimen untuk pemisahan file log, plot, sampel, dan checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed reprodusibilitas (default: {SEED})",
    )
    parser.add_argument(
        "--limit_batches",
        type=int,
        default=0,
        help="Batasi jumlah batch per epoch untuk uji cepat/debugging (0 = gunakan seluruh batch)",
    )
    parser.add_argument(
        "--compare_with",
        nargs="*",
        default=None,
        help="Daftar nama eksperimen lain (mis. --compare_with baseline) untuk di-plot gabungannya",
    )
    args = parser.parse_args()

    # Jika --lr diberikan, timpa lr_g dan lr_d
    if args.lr is not None:
        args.lr_g = args.lr
        args.lr_d = args.lr

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs dan --batch_size harus bernilai minimal 1.")
    if args.lr_g <= 0 or args.lr_d <= 0:
        parser.error("--lr_g dan --lr_d harus bernilai positif (> 0).")

    return args


# ---------------------------------------------------------------------------
# 2. Evaluasi Skor Keseimbangan Nash (Heuristik Checkpoint Terbaik)
# ---------------------------------------------------------------------------
def compute_equilibrium_gap(d_real_prob: float, d_fake_prob: float) -> float:
    """
    Menghitung jarak rata-rata probabilitas D(x) dan D(G(z)) terhadap titik ekuilibrium Nash (0.5).
    Semakin kecil nilai gap, semakin seimbang permainan minimax antara G dan D.
    """
    return abs(d_real_prob - 0.5) + abs(d_fake_prob - 0.5)


# ---------------------------------------------------------------------------
# 3. Pelatihan Satu Epoch (2-Step Adversarial Backpropagation)
# ---------------------------------------------------------------------------
def train_one_epoch(
    netG: nn.Module,
    netD: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optG: optim.Optimizer,
    optD: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    label_smoothing: bool = False,
    limit_batches: int = 0,
) -> Dict[str, float]:
    """
    Menjalankan satu epoch pelatihan cGAN dengan pembaruan gradien dua tahap:
        Tahap 1: Maksimalkan log(D(x|y)) + log(1 - D(G(z|y_fake)|y_fake)) -> Update bobot D.
        Tahap 2: Maksimalkan log(D(G(z|y_fake)|y_fake)) (non-saturating) -> Update bobot G.

    Returns:
        Dictionary rata-rata metrik epoch: {'d_loss', 'g_loss', 'd_real', 'd_fake'}.
    """
    netG.train()
    netD.train()

    real_target_val = REAL_LABEL_SMOOTH if label_smoothing else 1.0
    running = {"d_loss": 0.0, "g_loss": 0.0, "d_real": 0.0, "d_fake": 0.0}
    num_steps = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{total_epochs:02d}", leave=False)
    for step, (real_images, real_labels) in enumerate(progress_bar):
        if limit_batches > 0 and step >= limit_batches:
            break

        # Pindahkan batch data asli ke device komputasi
        real_images = real_images.to(device, non_blocking=True)  # [B, 1, 28, 28], rentang [-1, 1]
        real_labels = real_labels.to(device, non_blocking=True)  # [B], int64 (0..9)
        curr_batch_size = real_images.size(0)

        # Siapkan tensor target ground-truth untuk BCEWithLogitsLoss
        real_targets = torch.full((curr_batch_size, 1), real_target_val, device=device)  # [B, 1]
        fake_targets = torch.zeros((curr_batch_size, 1), device=device)                  # [B, 1]
        gen_targets = torch.ones((curr_batch_size, 1), device=device)                    # [B, 1]

        # Bangkitkan latent noise z dan label kelas acak untuk Generator
        z = torch.randn(curr_batch_size, LATENT_DIM, device=device)                      # [B, 100]
        fake_labels = torch.randint(0, NUM_CLASSES, (curr_batch_size,), device=device)   # [B]
        fake_images = netG(z, fake_labels)                                               # [B, 1, 28, 28]

        # =====================================================================
        # LANGKAH 1: Update Discriminator (netD)
        # Tujuan: Membedakan citra asli (target ~1) dan citra palsu (target 0).
        # Penting: `fake_images.detach()` memutus computation graph ke netG agar
        #          gradien loss_d TIDAK mengalir dan mengubah bobot Generator.
        # =====================================================================
        optD.zero_grad(set_to_none=True)

        real_logits = netD(real_images, real_labels)           # [B, 1] logit citra asli
        loss_d_real = criterion(real_logits, real_targets)     # scalar tensor

        fake_logits_d = netD(fake_images.detach(), fake_labels)  # [B, 1] logit citra palsu (detached)
        loss_d_fake = criterion(fake_logits_d, fake_targets)     # scalar tensor

        loss_d = 0.5 * (loss_d_real + loss_d_fake)             # [1] rata-rata loss real & fake
        loss_d.backward()                                      # Hitung d(loss_d)/d(theta_D)
        optD.step()                                            # Perbarui bobot Discriminator

        # =====================================================================
        # LANGKAH 2: Update Generator (netG)
        # Tujuan: Menipu Discriminator agar menilai citra palsu sebagai asli (target 1.0).
        # Penting: Gunakan `fake_images` TANPA `.detach()` supaya gradien dari output
        #          netD mengalir mundur menembus netD menuju parameter netG.
        # =====================================================================
        optG.zero_grad(set_to_none=True)

        fake_logits_g = netD(fake_images, fake_labels)         # [B, 1] evaluasi ulang oleh D terbaru
        loss_g = criterion(fake_logits_g, gen_targets)         # scalar tensor (non-saturating loss)
        loss_g.backward()                                      # Hitung d(loss_g)/d(theta_G)
        optG.step()                                            # Perbarui bobot Generator

        # Hitung probabilitas rata-rata D(x) dan D(G(z)) untuk pemantauan stabilitas
        with torch.no_grad():
            d_real_prob = torch.sigmoid(real_logits).mean().item()
            d_fake_prob = torch.sigmoid(fake_logits_d).mean().item()

        running["d_loss"] += loss_d.item()
        running["g_loss"] += loss_g.item()
        running["d_real"] += d_real_prob
        running["d_fake"] += d_fake_prob
        num_steps += 1

        if hasattr(progress_bar, "set_postfix"):
            progress_bar.set_postfix(
                D_loss=f"{loss_d.item():.4f}",
                G_loss=f"{loss_g.item():.4f}",
                Dx=f"{d_real_prob:.2f}",
                DGz=f"{d_fake_prob:.2f}",
            )

    if num_steps == 0:
        raise RuntimeError("DataLoader tidak menghasilkan batch untuk dilatih.")

    return {k: val / num_steps for k, val in running.items()}


# ---------------------------------------------------------------------------
# 4. Fungsi Utama Pelatihan & Manajemen Eksperimen
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_directories()

    # Siapkan jalur artefak spesifik eksperimen agar hasil tuning tidak saling menimpa
    exp_ckpt_dir: Path = CHECKPOINT_DIR / args.exp_name
    exp_sample_dir: Path = SAMPLE_DIR / args.exp_name
    exp_csv_path: Path = LOG_DIR / f"metrics_{args.exp_name}.csv"
    exp_plot_path: Path = PLOT_DIR / f"loss_curves_{args.exp_name}.png"
    exp_ckpt_dir.mkdir(parents=True, exist_ok=True)
    exp_sample_dir.mkdir(parents=True, exist_ok=True)

    # Bersihkan file CSV lama dengan nama eksperimen yang sama agar tidak terduplikasi
    if exp_csv_path.exists():
        exp_csv_path.unlink()

    # Inisialisasi DataLoader MNIST (kontrak Anggota 2)
    train_loader = get_dataloader(batch_size=args.batch_size, train=True)

    # Inisialisasi Model Generator & Discriminator beserta bobot awal Normal(0, 0.02) (kontrak Anggota 1)
    netG = ConditionalGenerator().to(DEVICE)
    netD = ConditionalDiscriminator().to(DEVICE)
    initialize_weights(netG)
    initialize_weights(netD)

    # Fungsi Loss Adversarial & Optimizer Adam dengan momentum (BETA1, BETA2)
    criterion = nn.BCEWithLogitsLoss()
    optG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(BETA1, BETA2))
    optD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(BETA1, BETA2))

    # Buat fixed_noise dan fixed_labels satu kali di awal agar perkembangan tiap epoch konsisten
    fixed_noise, fixed_labels = make_fixed_inputs(
        n_per_class=8, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, device=DEVICE
    )  # fixed_noise: [80, 100], fixed_labels: [80]

    print("=== PELATIHAN CONDITIONAL GAN MNIST (train.py) ===")
    print(f"Device Eksekusi    : {DEVICE} (Seed: {args.seed})")
    print(f"Nama Eksperimen    : {args.exp_name}")
    print(f"Hyperparameter     : Epochs={args.epochs} | Batch={args.batch_size} | "
          f"LR_G={args.lr_g:g} | LR_D={args.lr_d:g} | Betas=({BETA1}, {BETA2}) | "
          f"LabelSmoothing={args.label_smoothing}")
    print(f"Total Batch/Epoch  : {len(train_loader) if not args.limit_batches else min(len(train_loader), args.limit_batches)}")

    best_gap = float("inf")
    best_epoch = 1
    warmup_epochs = max(1, args.epochs // 5) if args.epochs > 2 else 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        metrics = train_one_epoch(
            netG=netG,
            netD=netD,
            dataloader=train_loader,
            optG=optG,
            optD=optD,
            criterion=criterion,
            device=DEVICE,
            epoch=epoch,
            total_epochs=args.epochs,
            label_smoothing=args.label_smoothing,
            limit_batches=args.limit_batches,
        )

        # 1. Catat metrik ke CSV eksperimen dan CSV utama (METRICS_CSV_PATH bila baseline)
        log_metrics_to_csv(
            filepath=exp_csv_path,
            epoch=epoch,
            d_loss=metrics["d_loss"],
            g_loss=metrics["g_loss"],
            lr_g=args.lr_g,
            lr_d=args.lr_d,
            batch_size=args.batch_size,
        )
        if args.exp_name == "baseline":
            if epoch == 1 and METRICS_CSV_PATH.exists():
                METRICS_CSV_PATH.unlink()
            log_metrics_to_csv(
                filepath=METRICS_CSV_PATH,
                epoch=epoch,
                d_loss=metrics["d_loss"],
                g_loss=metrics["g_loss"],
                lr_g=args.lr_g,
                lr_d=args.lr_d,
                batch_size=args.batch_size,
            )

        # 2. Simpan visualisasi sampel grid angka 0-9 setiap SAMPLE_INTERVAL epoch
        if epoch % SAMPLE_INTERVAL == 0 or epoch == args.epochs:
            save_sample_images(
                generator=netG,
                epoch=epoch,
                fixed_noise=fixed_noise,
                fixed_labels=fixed_labels,
                device=DEVICE,
                output_dir=exp_sample_dir,
            )

        # 3. Simpan checkpoint terakhir (last.pth) & perbarui BEST_GEN_PATH / BEST_DISC_PATH
        ckpt_state = {
            "epoch": epoch,
            "generator_state_dict": netG.state_dict(),
            "discriminator_state_dict": netD.state_dict(),
            "optimizer_G_state_dict": optG.state_dict(),
            "optimizer_D_state_dict": optD.state_dict(),
            "d_loss": float(metrics["d_loss"]),
            "g_loss": float(metrics["g_loss"]),
            "hparams": {
                "exp_name": args.exp_name,
                "lr_g": args.lr_g,
                "lr_d": args.lr_d,
                "batch_size": args.batch_size,
                "label_smoothing": args.label_smoothing,
            },
        }
        save_checkpoint(ckpt_state, exp_ckpt_dir / "last.pth")
        save_checkpoints(
            generator=netG,
            discriminator=netD,
            epoch=epoch,
            d_loss=metrics["d_loss"],
            g_loss=metrics["g_loss"],
            gen_path=BEST_GEN_PATH,
            disc_path=BEST_DISC_PATH,
        )

        # Simpan best.pth jika keseimbangan D(x) & D(G(z)) paling mendekati 0.5
        gap = compute_equilibrium_gap(metrics["d_real"], metrics["d_fake"])
        is_best = (epoch > warmup_epochs) and (gap < best_gap)
        if is_best or epoch == 1:
            if is_best:
                best_gap = gap
                best_epoch = epoch
            save_checkpoint(ckpt_state, exp_ckpt_dir / "best.pth")

        elapsed = time.time() - t0
        best_marker = " [BEST]" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Loss_D: {metrics['d_loss']:.4f} | Loss_G: {metrics['g_loss']:.4f} | "
            f"D(x): {metrics['d_real']:.3f} | D(G(z)): {metrics['d_fake']:.3f} | "
            f"{elapsed:.1f}s{best_marker}"
        )

    # 4. Buat grafik perbandingan Loss D vs Loss G di akhir pelatihan
    plot_loss_curves(csv_path=exp_csv_path, output_image_path=exp_plot_path)
    if args.exp_name == "baseline" and METRICS_CSV_PATH.exists():
        plot_loss_curves(csv_path=METRICS_CSV_PATH, output_image_path=PLOT_DIR / "loss_curves.png")

    # 5. Jika --compare_with diberikan, buat grafik gabungan beberapa CSV eksperimen
    if args.compare_with:
        compare_dict = {other: LOG_DIR / f"metrics_{other}.csv" for other in args.compare_with}
        compare_dict[args.exp_name] = exp_csv_path
        comp_plot_path = PLOT_DIR / f"compare_{'_vs_'.join(compare_dict.keys())}.png"
        plot_experiment_comparison(compare_dict, comp_plot_path)
        print(f"Grafik Gabungan    : {comp_plot_path}")

    total_minutes = (time.time() - start_time) / 60.0
    print("=== PELATIHAN SELESAI ===")
    print(f"Waktu Total        : {total_minutes:.2f} menit (Epoch Terbaik: {best_epoch})")
    print(f"Checkpoint Eksper. : {exp_ckpt_dir / 'best.pth'} & {exp_ckpt_dir / 'last.pth'}")
    print(f"Checkpoint Utama   : {BEST_GEN_PATH} (siap untuk generate.py)")
    print(f"Sampel Citra 0-9   : {exp_sample_dir}")
    print(f"Log CSV & Grafik   : {exp_csv_path} | {exp_plot_path}")


if __name__ == "__main__":
    main()

