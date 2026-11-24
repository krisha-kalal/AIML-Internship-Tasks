
import torch
import torch.nn as nn

class SceneCNN(nn.Module):
    def __init__(self, num_classes: int = 6, dropout_rate: float = 0.4):
        super(SceneCNN, self).__init__()

        # feature extractor: 4 convolutional blocks
        self.features = nn.Sequential(
            # block 1: 3 -> 32 channels | 150x150 -> 75x75
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 2: 32 -> 64 channels | 75x75 -> 37x37
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 3: 64 -> 128 channels | 37x37 -> 18x18
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # block 4: 128 -> 256 channels | 18x18 -> 9x9
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 9 * 9, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    sample_input = torch.randn(4, 3, 150, 150)
    model = SceneCNN(num_classes=6)
    output = model(sample_input)
    print(f"Architecture:\n{model}\n")
    print(f"Input shape: {sample_input.shape} -> Output shape: {output.shape}")
    print(f"Total trainable parameters: {count_trainable_parameters(model):,}")
