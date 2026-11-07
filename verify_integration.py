import torch
import torch.nn as nn

import config
import data
from models import ConditionalDiscriminator, ConditionalGenerator, initialize_weights


def run_integration_check() -> None:
    torch.manual_seed(config.SEED)
    config.ensure_directories()

    test_batch = 8

    if hasattr(data, "get_dataloader"):
        loader = data.get_dataloader(batch_size=test_batch, train=True)
        real_images, real_labels = next(iter(loader))
        real_images = real_images.to(config.DEVICE)
        real_labels = real_labels.to(config.DEVICE)
        data_status = "AKTIF (DataLoader Anggota 2 terdeteksi)"
    else:
        real_images = torch.empty(test_batch, *config.IMG_SHAPE, device=config.DEVICE).uniform_(-1.0, 1.0)
        real_labels = torch.randint(0, config.NUM_CLASSES, (test_batch,), device=config.DEVICE)
        data_status = "SIMULASI (data.py Anggota 2 masih kosong)"

    netG = ConditionalGenerator().to(config.DEVICE)
    netD = ConditionalDiscriminator().to(config.DEVICE)
    initialize_weights(netG)
    initialize_weights(netD)

    criterion = nn.BCEWithLogitsLoss()

    noise = torch.randn(test_batch, config.LATENT_DIM, device=config.DEVICE)
    fake_images = netG(noise, real_labels)

    real_logits = netD(real_images, real_labels)
    fake_logits = netD(fake_images.detach(), real_labels)

    real_targets = torch.ones_like(real_logits)
    fake_targets = torch.zeros_like(fake_logits)

    loss_d = 0.5 * (criterion(real_logits, real_targets) + criterion(fake_logits, fake_targets))
    loss_g = criterion(netD(fake_images, real_labels), real_targets)

    loss_d.backward()
    loss_g.backward()

    assert real_images.shape == (test_batch, config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)
    assert fake_images.shape == (test_batch, config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)
    assert real_logits.shape == (test_batch, 1)
    assert fake_logits.shape == (test_batch, 1)

    print("=== VERIFIKASI INTEGRASI MODUL cGAN MNIST ===")
    print(f"Device Eksekusi       : {config.DEVICE}")
    print(f"Status Modul data.py  : {data_status}")
    print(f"Shape Citra Real/Fake : {list(real_images.shape)} / {list(fake_images.shape)}")
    print(f"Rentang Piksel Fake   : [{fake_images.min().item():.3f}, {fake_images.max().item():.3f}]")
    print(f"Shape Output Logit D  : {list(real_logits.shape)}")
    print(f"Uji Hitung Loss D & G : Loss_D = {loss_d.item():.4f} | Loss_G = {loss_g.item():.4f}")
    print("Status Integrasi      : SELURUH KONTRAK SHAPE & BACKPROPAGATION VALID!")


if __name__ == "__main__":
    run_integration_check()