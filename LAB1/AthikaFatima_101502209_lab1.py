# cifar10_cnn.py
# PyTorch CIFAR-10: model + train + validate + plots
import os, math, time, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

# -----------------------
# 0) Repro & Device
# -----------------------
SEED = 42
random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 100

# -----------------------
# 1) Data
# -----------------------
DATA_DIR = "./data"
BATCH_SIZE = 128
NUM_WORKERS = min(8, os.cpu_count() or 2)

# CIFAR-10 stats (approx):
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tfms)
val_ds   = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

CLASSES = train_ds.classes  # ['airplane', 'automobile', ...]
NUM_CLASSES = len(CLASSES)

# -----------------------
# 2) Model
# -----------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, p=1, s=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=k, padding=p, stride=s, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.drop(x)
        return x

class CifarSmallCNN(nn.Module):
    """
    A compact, strong baseline (≈0.6–0.8M params) for CIFAR-10.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = ConvBlock(3, 64)
        self.block1 = nn.Sequential(
            ConvBlock(64, 64), ConvBlock(64, 64, dropout=0.1), nn.MaxPool2d(2)  # 32->16
        )
        self.block2 = nn.Sequential(
            ConvBlock(64, 128), ConvBlock(128, 128, dropout=0.1), nn.MaxPool2d(2) # 16->8
        )
        self.block3 = nn.Sequential(
            ConvBlock(128, 256), ConvBlock(256, 256, dropout=0.2), nn.MaxPool2d(2) # 8->4
        )
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head(x)        # (B, 256, 1, 1)
        x = x.flatten(1)        # (B, 256)
        return self.fc(x)

model = CifarSmallCNN(NUM_CLASSES).to(DEVICE)
# Use torch.compile if available (PyTorch 2.0+)
if hasattr(torch, "compile"):
    try:
        model = torch.compile(model)
    except Exception:
        pass

# -----------------------
# 3) Optimizer & Schedule
# -----------------------
EPOCHS = 30
LR = 0.05
WEIGHT_DECAY = 5e-4

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True, weight_decay=WEIGHT_DECAY)
# Cosine decay + warmup
def cosine_lr(step, total_steps, base_lr, warmup_steps=500):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

total_steps = EPOCHS * len(train_loader)
scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

# -----------------------
# 4) Utilities
# -----------------------
def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        with autocast(enabled=(DEVICE.type == "cuda")):
            logits = model(xb)
            loss = criterion(logits, yb)
        bs = yb.size(0)
        tot_loss += loss.item() * bs
        tot_acc  += (logits.argmax(1) == yb).sum().item()
        n += bs
    return tot_loss / n, tot_acc / n

def set_lr(optim, new_lr):
    for g in optim.param_groups: g["lr"] = new_lr

# -----------------------
# 5) Train Loop
# -----------------------
def train():
    train_losses, val_losses, val_accs = [], [], []
    criterion = nn.CrossEntropyLoss()

    step = 0
    best_val_acc = 0.0
    best_path = Path("best_cifar10_cnn.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        start = time.time()
        for i, (xb, yb) in enumerate(train_loader, 1):
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            lr_now = cosine_lr(step, total_steps, LR, warmup_steps=500)
            set_lr(optimizer, lr_now)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            step += 1
            if i % PRINT_EVERY == 0:
                print(f"Epoch {epoch:02d} | Step {i:04d}/{len(train_loader)} | "
                      f"LR {lr_now:.5f} | Loss {running/PRINT_EVERY:.4f}")
                train_losses.append(running / PRINT_EVERY)
                running = 0.0

        vloss, vacc = evaluate(model, val_loader)
        val_losses.append(vloss); val_accs.append(vacc)
        dur = time.time() - start
        print(f"Epoch {epoch:02d} done in {dur:.1f}s | Val Loss {vloss:.4f} | Val Acc {vacc*100:.2f}%")

        # Save best
        if vacc > best_val_acc:
            best_val_acc = vacc
            torch.save({"model": model.state_dict(),
                        "acc": best_val_acc,
                        "epoch": epoch,
                        "classes": CLASSES}, best_path)
            print(f"  ↳ New best! Saved to {best_path} (acc {best_val_acc*100:.2f}%)")

    # Final evaluation with best weights
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        vloss, vacc = evaluate(model, val_loader)
        print(f"\nBest checkpoint from epoch {ckpt['epoch']} | Val Acc {vacc*100:.2f}%")

    # Plots
    plot_curves(train_losses, val_losses, val_accs)

def plot_curves(train_losses, val_losses, val_accs):
    Path("artifacts").mkdir(exist_ok=True)
    # Training loss (per PRINT_EVERY batches)
    plt.figure()
    plt.plot(train_losses)
    plt.title("Training Loss (moving)")
    plt.xlabel(f"Updates (x{PRINT_EVERY} batches)"); plt.ylabel("Loss")
    plt.tight_layout(); plt.savefig("artifacts/training_loss.png"); plt.close()

    # Validation loss
    plt.figure()
    plt.plot(val_losses)
    plt.title("Validation Loss (per epoch)")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.tight_layout(); plt.savefig("artifacts/val_loss.png"); plt.close()

    # Validation accuracy
    plt.figure()
    plt.plot([a*100 for a in val_accs])
    plt.title("Validation Accuracy (per epoch)")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.tight_layout(); plt.savefig("artifacts/val_acc.png"); plt.close()

    print("Saved plots to artifacts/: training_loss.png, val_loss.png, val_acc.png")

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    train()
