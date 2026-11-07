import torch
import torch.nn as nn

from config import (
    DEVICE,
    DISC_BASE_CHANNELS,
    DISC_DROPOUT,
    GEN_BASE_CHANNELS,
    IMG_CHANNELS,
    IMG_SIZE,
    LABEL_EMBED_DIM,
    LATENT_DIM,
    NUM_CLASSES,
    SEED,
)


class ConditionalGenerator(nn.Module):
    

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = LABEL_EMBED_DIM,
        base_channels: int = GEN_BASE_CHANNELS,
        img_channels: int = IMG_CHANNELS,
    ) -> None:
        super().__init__()
        self.init_spatial_size = IMG_SIZE // 4  # 28 // 4 = 7
        self.base_channels = base_channels

        
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        
        combined_input_dim = latent_dim + embed_dim  
        self.feature_projection = nn.Sequential(
            nn.Linear(
                combined_input_dim,
                base_channels * self.init_spatial_size * self.init_spatial_size,
                bias=False,
            ),
            nn.BatchNorm1d(base_channels * self.init_spatial_size * self.init_spatial_size),
            nn.ReLU(inplace=True),
        )

        
        self.upsample_blocks = nn.Sequential(
            
            nn.ConvTranspose2d(
                in_channels=base_channels,
                out_channels=base_channels // 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(
                in_channels=base_channels // 2,
                out_channels=img_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.Tanh(),  
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
       
        label_emb = self.label_embedding(labels) 

        
        gen_input = torch.cat([z, label_emb], dim=1)  

        
        projected = self.feature_projection(gen_input)  
        feature_map = projected.view(
            -1, self.base_channels, self.init_spatial_size, self.init_spatial_size
        )  

        
        generated_img = self.upsample_blocks(feature_map) 
        return generated_img


class ConditionalDiscriminator(nn.Module):
    

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        img_channels: int = IMG_CHANNELS,
        img_size: int = IMG_SIZE,
        base_channels: int = DISC_BASE_CHANNELS,
        dropout_prob: float = DISC_DROPOUT,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        final_spatial_size = img_size // 4  # 28 // 4 = 7

        
        self.label_condition_map = nn.Embedding(num_classes, img_size * img_size)

        
        self.conv_blocks = nn.Sequential(
            
            nn.Conv2d(
                in_channels=img_channels + 1, 
                out_channels=base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(p=dropout_prob),
            
            nn.Conv2d(
                in_channels=base_channels,
                out_channels=base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(p=dropout_prob),
        )

        
        flattened_dim = (base_channels * 2) * final_spatial_size * final_spatial_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 1),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        
        batch_size = images.size(0)

        
        label_map = self.label_condition_map(labels)  
        label_map = label_map.view(batch_size, 1, self.img_size, self.img_size)  

        
        conditioned_input = torch.cat([images, label_map], dim=1)  

        
        features = self.conv_blocks(conditioned_input)  
        validity_logit = self.classifier(features) 
        return validity_logit


def initialize_weights(model: nn.Module) -> None:
   
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
            nn.init.constant_(m.bias.data, 0.0)


if __name__ == "__main__":
    
    torch.manual_seed(SEED)
    test_batch_size = 8

    
    netG = ConditionalGenerator().to(DEVICE)
    netD = ConditionalDiscriminator().to(DEVICE)
    initialize_weights(netG)
    initialize_weights(netD)

    
    dummy_z = torch.randn(test_batch_size, LATENT_DIM, device=DEVICE)
    dummy_labels = torch.randint(0, NUM_CLASSES, (test_batch_size,), device=DEVICE)

    
    fake_images = netG(dummy_z, dummy_labels)
    disc_logits = netD(fake_images, dummy_labels)

    
    params_g = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    params_d = sum(p.numel() for p in netD.parameters() if p.requires_grad)

    print("=== SANITY CHECK: models.py ===")
    print(f"Input Noise (z) Shape      : {list(dummy_z.shape)}")
    print(f"Input Labels Shape         : {list(dummy_labels.shape)}")
    print(f"Generator Output Shape     : {list(fake_images.shape)}  (Target: [{test_batch_size}, 1, 28, 28])")
    print(f"Generator Min/Max Range    : [{fake_images.min().item():.3f}, {fake_images.max().item():.3f}]")
    print(f"Discriminator Output Shape : {list(disc_logits.shape)}        (Target: [{test_batch_size}, 1])")
    print(f"Total Parameter Generator  : {params_g:,} parameter")
    print(f"Total Parameter Discrim.   : {params_d:,} parameter")

    assert fake_images.shape == (test_batch_size, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), "Shape G salah!"
    assert disc_logits.shape == (test_batch_size, 1), "Shape D salah!"
    print("Status                     : Arsitektur Generator & Discriminator VALID!")