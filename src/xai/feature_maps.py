"""
Feature Map Extraction and Intermediate Activation Visualization.
Uses PyTorch forward hooks to capture representations as data passes
through convolutional layers, revealing how features evolve with network depth.
"""

import io
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from src.data.dataset import pil_to_base64


class FeatureExtractor:
    """
    Hook manager to capture intermediate activation maps from convolutional layers.
    """
    def __init__(self, model: nn.Module, target_layers: Optional[List[str]] = None):
        self.model = model
        self.model.eval()
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        if target_layers is None:
            if hasattr(model, "get_visualizable_layers"):
                target_layers = model.get_visualizable_layers()
            else:
                target_layers = [name for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]

        self.target_layers = target_layers
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward hooks on the target layers."""
        for layer_name in self.target_layers:
            if hasattr(self.model, "get_layer_by_name"):
                layer_module = self.model.get_layer_by_name(layer_name)
            else:
                layer_module = dict(self.model.named_modules()).get(layer_name)

            if layer_module is not None:
                hook = layer_module.register_forward_hook(self._make_hook(layer_name))
                self.hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu()
        return hook

    def remove_hooks(self):
        """Clean up registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Performs forward pass and captures activations.
        Returns model output logits and dictionary of layer activations.
        """
        self.activations.clear()
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        logits = self.model(input_tensor)
        return logits, self.activations


def visualize_channel_map(
    activation: torch.Tensor,
    channel_idx: int,
    cmap: str = "viridis",
    upsample_size: Optional[Tuple[int, int]] = (128, 128)
) -> Image.Image:
    """
    Renders a single channel activation map as a high-contrast heatmap image.
    activation shape: (1, C, H, W) or (C, H, W).
    """
    if activation.ndim == 4:
        activation = activation.squeeze(0)

    num_channels = activation.shape[0]
    channel_idx = max(0, min(channel_idx, num_channels - 1))
    channel_act = activation[channel_idx].numpy()

    # Min-max normalization for visualization
    act_min, act_max = channel_act.min(), channel_act.max()
    if act_max > act_min:
        norm_act = (channel_act - act_min) / (act_max - act_min)
    else:
        norm_act = np.zeros_like(channel_act)

    colormap = matplotlib.colormaps.get_cmap(cmap)
    colored = colormap(norm_act)  # RGBA in [0, 1]
    colored_uint8 = (colored[:, :, :3] * 255.0).astype(np.uint8)

    img = Image.fromarray(colored_uint8)
    if upsample_size is not None:
        img = img.resize(upsample_size, Image.Resampling.NEAREST)
    return img


def generate_feature_grid(
    activation: torch.Tensor,
    max_channels: int = 16,
    cols: int = 4,
    cmap: str = "viridis"
) -> Image.Image:
    """
    Creates a composite grid image of the top activated channels in a layer.
    """
    if activation.ndim == 4:
        activation = activation.squeeze(0)

    num_channels = activation.shape[0]
    # Rank channels by mean activation energy
    channel_energies = [activation[c].abs().mean().item() for c in range(num_channels)]
    top_channel_indices = np.argsort(channel_energies)[::-1][:max_channels]

    rows = int(np.ceil(len(top_channel_indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2), facecolor="#0f172a")
    plt.subplots_adjust(wspace=0.15, hspace=0.3)

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.set_facecolor("#0f172a")
        if i < len(top_channel_indices):
            c_idx = int(top_channel_indices[i])
            c_act = activation[c_idx].numpy()
            c_min, c_max = c_act.min(), c_act.max()
            norm_act = (c_act - c_min) / (c_max - c_min) if c_max > c_min else np.zeros_like(c_act)

            im = ax.imshow(norm_act, cmap=cmap)
            ax.set_title(f"Ch #{c_idx}", color="#94a3b8", fontsize=9, pad=3)
        ax.axis("off")

    buffer = io.BytesIO()
    plt.savefig(buffer, format="PNG", bbox_inches="tight", dpi=100, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer)


def get_layer_statistics(activation: torch.Tensor) -> Dict[str, Any]:
    """Computes interpretability statistics for a layer's activations."""
    if activation.ndim == 4:
        activation = activation.squeeze(0)

    act_np = activation.numpy()
    total_elements = act_np.size
    active_elements = np.sum(act_np > 0)
    sparsity_ratio = 1.0 - (active_elements / max(total_elements, 1))

    channel_means = np.mean(act_np, axis=(1, 2))
    channel_maxes = np.max(act_np, axis=(1, 2))

    return {
        "num_channels": int(activation.shape[0]),
        "spatial_resolution": f"{activation.shape[1]}x{activation.shape[2]}",
        "mean_activation": float(np.mean(act_np)),
        "max_activation": float(np.max(act_np)),
        "sparsity_ratio": float(sparsity_ratio),
        "most_active_channel": int(np.argmax(channel_means)),
        "top_channel_means": [float(v) for v in channel_means[:8]],
        "top_channel_maxes": [float(v) for v in channel_maxes[:8]]
    }
