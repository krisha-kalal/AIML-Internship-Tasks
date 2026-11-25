import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

loader = DataLoader(
    datasets.ImageFolder("./data/seg_train/seg_train", transform=transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor()])),
    batch_size=8, shuffle=True
)

def imshow(img):
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.axis("off")
    plt.show()

images, labels = next(iter(loader))
print([loader.dataset.classes[i] for i in labels])
imshow(torchvision.utils.make_grid(images))

class CNN(nn.Module):
    def __init__(self, num_classes=6, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 9 * 9, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)

        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        return self.fc2(x)

if __name__ == "__main__":
    model = CNN()
    out = model(torch.randn(4, 3, 150, 150))
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Output shape: {out.shape} | Trainable params: {params:,}")
