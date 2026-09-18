"""
Filter & Kernel Visualization.
Extracts learned 2D/3D convolutional weights to inspect what spatial frequency,
orientation, and color features the network's filters have learned.
"""

import io
from typing import Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn


def extract_layer_weights(model: nn.Module, layer_name: str) -> torch.Tensor:
    """Extracts weights tensor of a given convolutional layer."""
    if hasattr(model, "get_layer_by_name"):
        layer = model.get_layer_by_name(layer_name)
    else:
        layer = dict(model.named_modules()).get(layer_name)

    if layer is None:
        raise ValueError(f"Layer {layer_name} not found in model.")
    if not hasattr(layer, "weight"):
        raise ValueError(f"Layer {layer_name} does not have weight attributes.")

    return layer.weight.detach().cpu()


def visualize_conv_filters(
    weights: torch.Tensor,
    max_filters: int = 32,
    cols: int = 8,
    cmap: str = "magma"
) -> Image.Image:
    """
    Renders convolutional filter kernels as an organized grid image.
    If in_channels == 3 (e.g. Conv1), renders full RGB color kernels.
    If in_channels > 3 (deeper layers), renders channel-collapsed or single slice kernels.
    """
    out_channels, in_channels, kh, kw = weights.shape
    num_filters = min(out_channels, max_filters)
    weights = weights[:num_filters]

    rows = int(np.ceil(num_filters / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5), facecolor="#0f172a")
    plt.subplots_adjust(wspace=0.1, hspace=0.25)

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.set_facecolor("#0f172a")
        if i < num_filters:
            filt = weights[i].numpy()

            if in_channels == 3:
                # RGB filter (normalized per filter across channels)
                f_min, f_max = filt.min(), filt.max()
                filt_norm = (filt - f_min) / (f_max - f_min) if f_max > f_min else np.zeros_like(filt)
                filt_rgb = np.transpose(filt_norm, (1, 2, 0))
                ax.imshow(filt_rgb, interpolation="nearest")
            else:
                # Collapse over input channels (L2 norm or mean)
                filt_2d = np.linalg.norm(filt, axis=0)
                f_min, f_max = filt_2d.min(), filt_2d.max()
                filt_norm = (filt_2d - f_min) / (f_max - f_min) if f_max > f_min else np.zeros_like(filt_2d)
                ax.imshow(filt_norm, cmap=cmap, interpolation="nearest")

            ax.set_title(f"F#{i}", color="#94a3b8", fontsize=8, pad=2)
        ax.axis("off")

    buffer = io.BytesIO()
    plt.savefig(buffer, format="PNG", bbox_inches="tight", dpi=110, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer)
