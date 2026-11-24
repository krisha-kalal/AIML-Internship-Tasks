
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

def get_loaders(train_dir, test_dir, train_tf, test_tf, batch_size=32, val_split=0.15):
    # load training folder twice (with and without augmentations)
    full_train = datasets.ImageFolder(train_dir, transform=train_tf)
    full_val = datasets.ImageFolder(train_dir, transform=test_tf)
    test_data = datasets.ImageFolder(test_dir, transform=test_tf)

    # calculate 85% train / 15% validation sizes
    val_size = int(val_split * len(full_train))
    train_size = len(full_train) - val_size

    # use the same seed 42 so both splits pick the exact same images
    train_data, _ = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    _, val_data = random_split(full_val, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, full_train.classes
