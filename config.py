from pathlib import Path
import torch

#Konfigurasi
SEED: int = 42
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_WORKERS: int = 0

#Spek dataset MNIST 
IMG_SIZE: int = 28
IMG_CHANNELS: int = 1
IMG_SHAPE: tuple = (IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
NUM_CLASSES: int = 10

NORM_MEAN: tuple = (0.5,)
NORM_STD: tuple = (0.5,)

# Parameter Model cGAN 

LATENT_DIM: int = 100
LABEL_EMBED_DIM: int = 50
GEN_BASE_CHANNELS: int = 128
DISC_BASE_CHANNELS: int = 64
DISC_DROPOUT: float = 0.25

# Hyperparameter

BATCH_SIZE: int = 64
EPOCHS: int = 30
LR_G: float = 2e-4
LR_D: float = 2e-4
BETA1: float = 0.5
BETA2: float = 0.999
SAMPLE_INTERVAL: int = 1

# Jalur Direktori

BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "dataset"
OUTPUT_DIR: Path = BASE_DIR / "outputs"
CHECKPOINT_DIR: Path = OUTPUT_DIR / "checkpoints"
SAMPLE_DIR: Path = OUTPUT_DIR / "samples"
PLOT_DIR: Path = OUTPUT_DIR / "plots"
LOG_DIR: Path = OUTPUT_DIR / "logs"

BEST_GEN_PATH: Path = CHECKPOINT_DIR / "generator_latest.pth"
BEST_DISC_PATH: Path = CHECKPOINT_DIR / "discriminator_latest.pth"
METRICS_CSV_PATH: Path = LOG_DIR / "training_metrics.csv"
def ensure_directories() -> None:
    """Memastikan seluruh direktori dataset dan outputs tersedia sebelum program berjalan."""
    target_dirs = [DATA_DIR, CHECKPOINT_DIR, SAMPLE_DIR, PLOT_DIR, LOG_DIR]
    for directory in target_dirs:
        directory.mkdir(parents=True, exist_ok=True)
if __name__ == "__main__":
    
    ensure_directories()
    print("=== SANITY CHECK: config.py ===")
    print(f"Base Directory   : {BASE_DIR}")
    print(f"Compute Device   : {DEVICE} (Seed: {SEED})")
    print(f"Target Img Shape : [B, {IMG_CHANNELS}, {IMG_SIZE}, {IMG_SIZE}] | Kelas: {NUM_CLASSES}")
    print(f"Generator Input  : z=[B, {LATENT_DIM}] + label_emb=[B, {LABEL_EMBED_DIM}]")
    print(f"Hyperparameters  : Batch={BATCH_SIZE}, Epochs={EPOCHS}, LR_G={LR_G}, LR_D={LR_D}")
    print("Status           : Seluruh direktori output dan konfigurasi siap digunakan!")