
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def build_resnet18(num_classes: int = 6, freeze_backbone: bool = True, dropout_rate: float = 0.3) -> nn.Module:
    # loads pretrained ResNet-18 and replaces the final classification head.
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # replace final fully connected layer (512 -> num_classes)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )

    return model

def unfreeze_for_finetuning(model: nn.Module, unfreeze_layer3: bool = False):
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        elif unfreeze_layer3 and "layer3" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def inspect_layer_status(model: nn.Module):
    trainable = 0
    frozen = 0
    for name, module in model.named_children():
        params = list(module.parameters())
        if not params:
            continue
        is_trainable = any(p.requires_grad for p in params)
        num_p = sum(p.numel() for p in params)
        status = "TRAINABLE" if is_trainable else "FROZEN"
        print(f"Block: {name:<10} | Status: {status:<10} | Params: {num_p:,}")
        if is_trainable:
            trainable += num_p
        else:
            frozen += num_p
    print(f"Total Trainable: {trainable:,} | Total Frozen: {frozen:,}\n")

if __name__ == "__main__":
    model = build_resnet18(num_classes=6, freeze_backbone=True)
    print("Stage 1 (Feature Extraction):")
    inspect_layer_status(model)

    unfreeze_for_finetuning(model, unfreeze_layer3=False)
    print("Stage 2 (Fine-Tuning layer4 + fc):")
    inspect_layer_status(model)
