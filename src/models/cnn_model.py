"""
Deep Visual Net: An Interpretable Convolutional Architecture for XAI.
Designed with distinct hierarchical stages to enable rich feature visualization:
- Stage 1 (conv1): Low-level visual primitives (edges, color gradients)
- Stage 2 (conv2): Textures, corners, and simple geometric patterns
- Stage 3 (conv3): Mid-level component parts and motif assemblies
- Stage 4 (conv4): High-level semantic structures and category-specific patterns
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    """Standard Convolutional Block with Conv2d, BatchNorm, and LeakyReLU."""
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class DeepVisualNet(nn.Module):
    """
    4-Stage Interpretable Convolutional Neural Network.
    Exposes named stages for clean activation hook attachment.
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes

        # Layer 1: Edge & Low-level feature extraction
        self.conv1 = ConvBlock(in_channels, 32, pool=True)

        # Layer 2: Texture & Corner extraction
        self.conv2 = ConvBlock(32, 64, pool=True)

        # Layer 3: Complex motif & Object part extraction
        self.conv3 = ConvBlock(64, 128, pool=True)

        # Layer 4: High-level semantic representation
        self.conv4 = ConvBlock(128, 256, pool=False)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        logits = self.classifier(x)
        return logits

    def get_layer_by_name(self, layer_name: str) -> nn.Module:
        """Helper to retrieve a specific sub-layer for hooks."""
        layer_dict = {
            "conv1": self.conv1.conv,
            "conv1_block": self.conv1,
            "conv2": self.conv2.conv,
            "conv2_block": self.conv2,
            "conv3": self.conv3.conv,
            "conv3_block": self.conv3,
            "conv4": self.conv4.conv,
            "conv4_block": self.conv4,
        }
        if layer_name not in layer_dict:
            raise ValueError(f"Unknown layer: {layer_name}. Available: {list(layer_dict.keys())}")
        return layer_dict[layer_name]

    def get_visualizable_layers(self) -> list[str]:
        """Returns list of layers supported for feature map visualization."""
        return ["conv1", "conv2", "conv3", "conv4"]


def build_model(model_name: str = "deep_visual_net", num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Factory function for instantiating model architectures."""
    if model_name.lower() == "deep_visual_net":
        model = DeepVisualNet(num_classes=num_classes)
        return model
    elif model_name.lower() == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported model: {model_name}. Options: 'deep_visual_net', 'resnet18'")
