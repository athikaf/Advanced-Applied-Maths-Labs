import os
import math
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# ----------------------------
# Models
# ----------------------------

class ConvAE(nn.Module):
    """
    Simple convolutional Autoencoder for 28x28 grayscale MNIST.
    Latent is a small feature map; not meant for sampling.
    """
    def __init__(self, latent_channels: int = 16):
        super().__init__()
        # Encoder: [B,1,28,28] -> [B,latent,7,7]
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, 3, padding=1), # keep 7x7
        )
        # Decoder: mirror via transposed convs
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   # 28x28
            nn.Sigmoid()  # output in [0,1]
        )

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat


class ConvVAE(nn.Module):
    """
    Convolutional VAE for 28x28 MNIST.
    Latent is a vector (z_dim), suitable for sampling z~N(0,I) to generate.
    """
    def __init__(self, z_dim: int = 16):
        super().__init__()
        self.z_dim = z_dim

        # Encoder -> feature map 7x7x64, then to mu/logvar
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 7x7
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()      # 7*7*64 = 3136
        self.fc_mu = nn.Linear(3136, z_dim)
        self.fc_logvar = nn.Linear(3136, z_dim)

        # Decoder: z -> 7x7x64 -> upsample to 28x28
        self.fc_dec = nn.Linear(z_dim, 3136)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   # 28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x)
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 64, 7, 7)
        x_hat = self.dec(h)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

# ----------------------------
# Losses
# ----------------------------

def ae_loss(x_hat, x):
    # BCE is common for MNIST pixels in [0,1]
    return F.binary_cross_entropy(x_hat, x, reduction="mean")

def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    # Reconstruction term (Bernoulli likelihood -> BCE)
    rec = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
    # KL divergence to N(0,I):  -0.5 * sum(1 + logσ² − μ² − σ²)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return rec + beta * kld, rec, kld

# ----------------------------
# Training / Eval utils
# ----------------------------

def get_loaders(batch_size=128, data_root="./data", augment=False):
    tfms = [transforms.ToTensor()]  # values in [0,1]
    if augment:
        tfms = [
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ]
    transform = transforms.Compose(tfms)

    train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_set  = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

@torch.no_grad()
def save_reconstructions(model, loader, device, save_path="recon_grid.png", is_vae=False):
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(device)
    if is_vae:
        x_hat, _, _ = model(x)
    else:
        x_hat = model(x)
    grid = utils.make_grid(torch.cat([x[:8], x_hat[:8]], dim=0), nrow=8)
    utils.save_image(grid, save_path)
    print(f"Saved reconstruction grid to {save_path}")

@torch.no_grad()
def generate_samples_vae(model, n=64, device="cpu", outdir="synthetic_mnist", grid_path="samples_grid.png"):
    """
    Sample z~N(0,I) and decode to digits.
    Also saves individual PNGs and a tensor dataset file for reuse.
    """
    model.eval()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    z = torch.randn(n, model.z_dim, device=device)
    x_gen = model.decode(z).cpu()  # [n,1,28,28], in [0,1]

    # Save a grid preview
    grid = utils.make_grid(x_gen, nrow=int(math.sqrt(n)))
    utils.save_image(grid, os.path.join(outdir, grid_path))
    print(f"Saved sample grid to {os.path.join(outdir, grid_path)}")

    # Save individual images
    for i in range(n):
        utils.save_image(x_gen[i], os.path.join(outdir, f"sample_{i:04d}.png"))

    # Optionally save as a tensor file (for building a synthetic dataset later)
    torch.save(x_gen, os.path.join(outdir, "synthetic_tensor.pt"))
    print(f"Saved {n} synthetic digits as PNG and tensor to {outdir}")

def train_ae(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader, test_loader = get_loaders(args.batch_size, args.data, augment=args.augment)

    model = ConvAE(latent_channels=args.latent_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            x_hat = model(x)
            loss = ae_loss(x_hat, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)

        avg = total / len(train_loader.dataset)
        print(f"[AE] Epoch {epoch}/{args.epochs}  train_bce: {avg:.4f}")

        # quick recon preview each epoch
        save_reconstructions(model, test_loader, device, save_path=f"ae_recon_epoch{epoch}.png", is_vae=False)

    # Final recon
    save_reconstructions(model, test_loader, device, save_path="ae_recon_final.png", is_vae=False)
    torch.save(model.state_dict(), "ae_mnist.pt")
    print("Saved AE model to ae_mnist.pt")

def train_vae(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader, test_loader = get_loaders(args.batch_size, args.data, augment=args.augment)

    model = ConvVAE(z_dim=args.z_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, total_rec, total_kld = 0.0, 0.0, 0.0
        for x, _ in train_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, rec, kld = vae_loss(x_hat, x, mu, logvar, beta=args.beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bsz = x.size(0)
            total += loss.item() * bsz
            total_rec += rec.item() * bsz
            total_kld += kld.item() * bsz

        ntrain = len(train_loader.dataset)
        print(f"[VAE] Epoch {epoch}/{args.epochs}  loss: {total/ntrain:.4f}  rec: {total_rec/ntrain:.4f}  kld: {total_kld/ntrain:.4f}")

        # quick recon preview and sample preview each epoch
        save_reconstructions(model, test_loader, device, save_path=f"vae_recon_epoch{epoch}.png", is_vae=True)
        generate_samples_vae(model, n=min(64, args.generate if args.generate>0 else 64), device=device,
                             outdir=f"samples_epoch{epoch}", grid_path="grid.png")

    # Final artifacts
    save_reconstructions(model, test_loader, device, save_path="vae_recon_final.png", is_vae=True)
    if args.generate > 0:
        generate_samples_vae(model, n=args.generate, device=device, outdir=args.outdir, grid_path="grid.png")
    torch.save(model.state_dict(), "vae_mnist.pt")
    print("Saved VAE model to vae_mnist.pt")

# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Autoencoders for MNIST (AE + VAE for generation)")
    p.add_argument("--model", choices=["ae", "vae"], default="vae", help="Which model to train")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--data", type=str, default="./data", help="Where to store/download MNIST")
    p.add_argument("--augment", action="store_true", help="Use light augmentation (rotation)")

    # AE specific
    p.add_argument("--latent_ch", type=int, default=16, help="AE latent channels")

    # VAE specific
    p.add_argument("--z_dim", type=int, default=16, help="VAE latent dimension")
    p.add_argument("--beta", type=float, default=1.0, help="VAE beta (KL weight)")

    # Generation
    p.add_argument("--generate", type=int, default=0, help="Number of synthetic images to generate with VAE")
    p.add_argument("--outdir", type=str, default="synthetic_mnist", help="Output folder for synthetic images")
    return p.parse_args()

def main():
    args = parse_args()
    if args.model == "ae":
        train_ae(args)
    else:
        train_vae(args)

if __name__ == "__main__":
    main()
