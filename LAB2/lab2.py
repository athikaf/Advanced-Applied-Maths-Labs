# lab2_vgg_cifar.py
import argparse, math, time, os
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_loaders(batch_size=128, num_workers=4, use_cifar100=False):
    # CIFAR is 32x32; VGG expects 224x224. We resize + normalize to ImageNet stats.
    train_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    root = "./data"
    if use_cifar100:
        train_ds = datasets.CIFAR100(root, train=True,  download=True, transform=train_tfms)
        test_ds  = datasets.CIFAR100(root, train=False, download=True, transform=test_tfms)
    else:
        train_ds = datasets.CIFAR10(root, train=True,  download=True, transform=train_tfms)
        test_ds  = datasets.CIFAR10(root, train=False, download=True, transform=test_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, len(train_ds.classes), train_ds.classes

def replace_classifier_vgg(vgg, num_classes, head="simple"):
    in_feats = vgg.classifier[-1].in_features
    if head == "simple":
        # Classic single linear layer
        vgg.classifier[-1] = nn.Linear(in_feats, num_classes)
    elif head == "rich":
        # A slightly richer head for fine-tuning
        vgg.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )
    return vgg

def set_trainable(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def build_model(arch="vgg16", num_classes=10, mode=1):
    # Load pretrained VGG
    if arch == "vgg19":
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    else:
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # 1) Freeze all (feature extractor)
    if mode == 1:
        set_trainable(vgg.features, False)
        set_trainable(vgg.classifier, False)
        vgg = replace_classifier_vgg(vgg, num_classes, head="simple")
        set_trainable(vgg.classifier, True)  # only top head trainable

    # 2) Partial unfreeze (unfreeze last conv block)
    elif mode == 2:
        set_trainable(vgg.features, False)
        # Unfreeze last conv block (features indices differ slightly; last block ~ 24+ for VGG16)
        # We'll unfreeze last 5 conv layers conservatively.
        last_layers = list(vgg.features.children())[-7:]  # conv+relu+conv+relu+pool-ish
        for layer in last_layers:
            set_trainable(layer, True)
        vgg = replace_classifier_vgg(vgg, num_classes, head="simple")
        set_trainable(vgg.classifier, True)

    # 3) Fine-tune + richer head + full features trainable (optionally low LR)
    elif mode == 3:
        set_trainable(vgg.features, True)
        vgg = replace_classifier_vgg(vgg, num_classes, head="rich")
        set_trainable(vgg.classifier, True)

    return vgg

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, tot_correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        tot_loss += loss.item() * yb.size(0)
        tot_correct += (logits.argmax(1) == yb).sum().item()
        n += yb.size(0)
    return tot_loss / n, tot_correct / n

def train_one(model, train_loader, val_loader, epochs=10, lr=1e-3, wd=1e-4):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # Different LR for classifier vs features (common for transfer learning)
    params = [
        {"params": model.features.parameters(), "lr": lr * 0.1},
        {"params": model.classifier.parameters(), "lr": lr},
    ]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    best_acc, best_path = 0.0, "best_vgg_cifar.pt"

    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        t0 = time.time()
        for i, (xb, yb) in enumerate(train_loader, 1):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if i % 50 == 0:
                print(f"Epoch {epoch:02d} | Batch {i}/{len(train_loader)} | Loss {running/50:.4f}")
                running = 0.0

        vloss, vacc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch:02d} done in {time.time()-t0:.1f}s | Val Loss {vloss:.4f} | Val Acc {vacc*100:.2f}%")

        if vacc > best_acc:
            best_acc = vacc
            torch.save({"model": model.state_dict(), "acc": best_acc, "epoch": epoch}, best_path)
            print(f"  ↳ New best saved to {best_path} ({best_acc*100:.2f}%)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["vgg16", "vgg19"], default="vgg16")
    ap.add_argument("--mode", type=int, choices=[1,2,3], default=1,
                    help="1=freeze all; 2=partial unfreeze; 3=fine-tune+rich head")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--cifar100", action="store_true", help="use CIFAR-100 instead of CIFAR-10")
    args = ap.parse_args()

    print(f"Device: {DEVICE} | Arch: {args.arch} | Mode: {args.mode} | CIFAR100: {args.cifar100}")
    train_loader, val_loader, num_classes, classes = make_loaders(
        batch_size=args.batch, num_workers=min(8, os.cpu_count() or 2),
        use_cifar100=args.cifar100
    )
    model = build_model(args.arch, num_classes=num_classes, mode=args.mode)
    train_one(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, wd=args.wd)

if __name__ == "__main__":
    main()
