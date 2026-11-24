
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

PROJECT_ROOT = "/content/drive/MyDrive/intel-image-classifier"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocessing import get_transforms
from data_loader import get_loaders
from cnn_model import SceneCNN, count_trainable_parameters

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total

def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'], label='Val Loss')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()

    ax2.plot(epochs, history['train_acc'], label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], label='Val Accuracy')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dir = os.path.join(PROJECT_ROOT, "data/seg_train/seg_train")
    test_dir = os.path.join(PROJECT_ROOT, "data/seg_test/seg_test")

    if not os.path.exists(train_dir) and os.path.exists("/content/intel_data/seg_train/seg_train"):
        train_dir = "/content/intel_data/seg_train/seg_train"
        test_dir = "/content/intel_data/seg_test/seg_test"

    models_dir = os.path.join(PROJECT_ROOT, "models")
    plots_dir = os.path.join(PROJECT_ROOT, "outputs/plots")
    metrics_dir = os.path.join(PROJECT_ROOT, "outputs/metrics")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # hyperparameters
    IMG_SIZE = 150
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # data setup using your exact functions
    train_tf, test_tf = get_transforms(img_size=IMG_SIZE)
    train_loader, val_loader, _, class_names = get_loaders(
        train_dir, test_dir, train_tf, test_tf, batch_size=BATCH_SIZE
    )

    # model, loss, optimizer
    model = SceneCNN(num_classes=len(class_names), dropout_rate=0.4).to(device)
    print(f"Trainable Parameters: {count_trainable_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'img_size': IMG_SIZE
            }, os.path.join(models_dir, "cnn_best.pth"))

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s. Best Validation Accuracy: {best_val_acc:.4f}")

    with open(os.path.join(metrics_dir, "cnn_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    plot_curves(history, os.path.join(plots_dir, "cnn_training_curves.png"))

if __name__ == "__main__":
    main()
