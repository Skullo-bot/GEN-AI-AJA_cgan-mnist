import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import config
from models import ConditionalGenerator, initialize_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferensi Conditional GAN MNIST (Angka 0-9)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(config.BEST_GEN_PATH),
        help="Jalur file bobot generator (.pth)",
    )
    parser.add_argument(
        "--digit",
        type=int,
        default=None,
        choices=range(0, config.NUM_CLASSES),
        help="Angka  (0-9) yang ingin dibuat. Kosongkan untuk membuat semua angka dari 0-9.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Jumlah variasi baris sampel yang dihasilkan",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_grid.png",
        help="Nama file gambar hasil generate di dalam folder outputs/samples/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="Seed angka acak untuk reprodusibilitas inferensi",
    )
    return parser.parse_args()


def load_generator(checkpoint_path: Path, device: torch.device) -> tuple[ConditionalGenerator, str]:
    netG = ConditionalGenerator().to(device)
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(state_dict, dict) and "generator_state_dict" in state_dict:
            netG.load_state_dict(state_dict["generator_state_dict"])
        else:
            netG.load_state_dict(state_dict)
        status = f"Bobot terlatih dimuat dari: {checkpoint_path.name}"
    else:
        initialize_weights(netG)
        status = "Checkpoint belum ditemukan (menggunakan bobot awal untuk uji fungsi)"
    netG.eval()
    return netG, status


@torch.no_grad()
def generate_samples() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    config.ensure_directories()

    checkpoint_path = Path(args.checkpoint)
    netG, model_status = load_generator(checkpoint_path, config.DEVICE)

    if args.digit is not None:
        target_digits = [args.digit]
    else:
        target_digits = list(range(config.NUM_CLASSES))

    num_cols = len(target_digits)
    num_rows = max(1, args.rows)
    total_samples = num_rows * num_cols

    labels_list = target_digits * num_rows
    labels = torch.tensor(labels_list, dtype=torch.long, device=config.DEVICE)
    z = torch.randn(total_samples, config.LATENT_DIM, device=config.DEVICE)

    fake_images = netG(z, labels)
    fake_images = (fake_images.clamp(-1.0, 1.0) + 1.0) / 2.0
    fake_images_np = fake_images.squeeze(1).cpu().numpy()

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.6, num_rows * 1.6))

    for r in range(num_rows):
        for c in range(num_cols):
            idx = r * num_cols + c
            if num_rows == 1 and num_cols == 1:
                ax = axes
            elif num_rows == 1:
                ax = axes[c]
            elif num_cols == 1:
                ax = axes[r]
            else:
                ax = axes[r, c]

            ax.imshow(fake_images_np[idx], cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"Digit {target_digits[c]}", fontsize=10, fontweight="bold")

    plt.tight_layout()
    save_path = config.SAMPLE_DIR / args.output
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("=== INFERENSI CONDITIONAL GAN MNIST (generate.py) ===")
    print(f"Status Model     : {model_status}")
    print(f"Target Digit     : {target_digits}")
    print(f"Total Citra      : {total_samples} citra ({num_rows} baris x {num_cols} kolom)")
    print(f"Shape Output G   : {list(fake_images.shape)}")
    print(f"Lokasi Tersimpan : {save_path}")


if __name__ == "__main__":
    generate_samples()