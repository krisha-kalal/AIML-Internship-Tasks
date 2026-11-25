from google.colab import drive
drive.mount('/content/drive')

import os, sys, json, time, shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
import matplotlib.pyplot as plt

PROJECT_ROOT = "/content/drive/MyDrive/intel-image-classifier"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import get_transforms
from cnn_model import CNN

# Remove any half-copied local folder from earlier
if os.path.exists("/content/local_data"):
    shutil.rmtree("/content/local_data")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_balanced_subset(dataset, per_class, label_name):
    """Loads a balanced per-class subset quickly (takes ~8-12s even over Drive)."""
    t0 = time.time()
    counts = {c: 0 for c in range(len(dataset.classes))}
    imgs, lbls = [], []
    # Shuffle indices deterministically
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(dataset), generator=g).tolist()

    for idx in perm:
        _, target = dataset.samples[idx]  # Instant metadata check without opening file!
        if counts[target] < per_class:
            try:
                img, lbl = dataset[idx]
                imgs.append(img)
                lbls.append(lbl)
                counts[target] += 1
            except Exception:
                continue
        if all(v >= per_class for v in counts.values()):
            break

    print(f"Loaded {len(imgs)} {label_name} images ({per_class}/class) in {time.time() - t0:.1f}s")
    return TensorDataset(torch.stack(imgs), torch.tensor(lbls))

def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'], label='Val Loss')
    ax1.set_title('Loss Curve'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()

    ax2.plot(epochs, history['train_acc'], label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], label='Val Accuracy')
    ax2.set_title('Accuracy Curve'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dir = os.path.join(PROJECT_ROOT, "data/seg_train/seg_train")
    test_dir = os.path.join(PROJECT_ROOT, "data/seg_test/seg_test")
    for p in ["/content/data", "/content/intel_data"]:
        if os.path.exists(f"{p}/seg_train/seg_train"):
            train_dir, test_dir = f"{p}/seg_train/seg_train", f"{p}/seg_test/seg_test"
            break

    models_dir = os.path.join(PROJECT_ROOT, "models")
    plots_dir = os.path.join(PROJECT_ROOT, "outputs/plots")
    metrics_dir = os.path.join(PROJECT_ROOT, "outputs/metrics")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    IMG_SIZE = 150
    BATCH_SIZE = 64
    EPOCHS = 5
    train_tf, test_tf = get_transforms(img_size=IMG_SIZE)

    full_train = datasets.ImageFolder(train_dir, transform=train_tf)
    full_val = datasets.ImageFolder(test_dir, transform=test_tf)
    class_names = full_train.classes

    # 100 images/class train (600) + 30 images/class val (180) -> loads in ~10s, trains in ~10s!
    train_ds = load_balanced_subset(full_train, per_class=100, label_name="train")
    val_ds = load_balanced_subset(full_val, per_class=30, label_name="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CNN(num_classes=len(class_names), dropout=0.4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss, t_corr, t_tot = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * imgs.size(0)
            t_corr += (out.argmax(1) == lbls).sum().item()
            t_tot += lbls.size(0)

        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = criterion(out, lbls)
                v_loss += loss.item() * imgs.size(0)
                v_corr += (out.argmax(1) == lbls).sum().item()
                v_tot += lbls.size(0)

        train_loss, train_acc = t_loss / t_tot, t_corr / t_tot
        val_loss, val_acc = v_loss / v_tot, v_corr / v_tot
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'img_size': IMG_SIZE
            }, os.path.join(models_dir, "cnn_best.pth"))

    print(f"\nDONE in {time.time() - t_start:.1f}s! Best Val Acc: {best_val_acc:.4f}")
    with open(os.path.join(metrics_dir, "cnn_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    plot_curves(history, os.path.join(plots_dir, "cnn_training_curves.png"))

if __name__ == "__main__":
    main()
