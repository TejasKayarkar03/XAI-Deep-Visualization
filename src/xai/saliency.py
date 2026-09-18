"""
Saliency Map Generation: Vanilla Backprop and SmoothGrad.
Extracts pixel-level sensitivity gradients showing which individual input pixels
most strongly influence the model's classification score.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_vanilla_saliency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Computes vanilla backprop saliency map: |dy_c / dx|.
    Returns 2D saliency map normalized to [0, 1].
    """
    model.eval()
    device = next(model.parameters()).device
    x = input_tensor.clone().detach().to(device)
    x.requires_grad = True

    logits = model(x)
    if target_class is None:
        target_class = int(torch.argmax(logits, dim=1).item())

    score = logits[0, target_class]
    model.zero_grad()
    score.backward()

    # Gradients with respect to input: (1, C, H, W)
    grads = x.grad.detach().cpu().squeeze(0)  # (C, H, W)

    # Take maximum magnitude across color channels
    saliency, _ = torch.max(grads.abs(), dim=0)
    saliency = saliency.numpy()

    # Min-max normalization
    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min:
        saliency = (saliency - s_min) / (s_max - s_min)
    else:
        saliency = np.zeros_like(saliency)

    return saliency, target_class


def compute_smoothgrad_saliency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    num_samples: int = 15,
    noise_level: float = 0.15
) -> Tuple[np.ndarray, int]:
    """
    SmoothGrad: Averages saliency maps over noisy perturbations of the input
    to remove high-frequency visual noise and isolate stable attribution.
    """
    model.eval()
    device = next(model.parameters()).device
    x_base = input_tensor.clone().detach().to(device)

    if target_class is None:
        with torch.no_grad():
            logits = model(x_base)
            target_class = int(torch.argmax(logits, dim=1).item())

    stdev = noise_level * (x_base.max() - x_base.min()).item()
    accumulated_grads = torch.zeros_like(x_base[0]).cpu()

    for _ in range(num_samples):
        noise = torch.randn_like(x_base) * stdev
        x_noisy = (x_base + noise).clone().detach().requires_grad_(True)

        model.zero_grad()
        logits = model(x_noisy)
        score = logits[0, target_class]
        score.backward()

        accumulated_grads += x_noisy.grad.detach().cpu().squeeze(0).abs()

    avg_grads = accumulated_grads / num_samples
    saliency, _ = torch.max(avg_grads, dim=0)
    saliency = saliency.numpy()

    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min:
        saliency = (saliency - s_min) / (s_max - s_min)
    else:
        saliency = np.zeros_like(saliency)

    return saliency, target_class



def saliency_to_image(saliency: np.ndarray, cmap: str = "hot") -> Image.Image:
    """Converts a 2D saliency numpy array to a colored PIL Image."""
    colormap = matplotlib.colormaps.get_cmap(cmap)
    colored = colormap(saliency)[:, :, :3]
    uint8_img = (colored * 255.0).astype(np.uint8)
    return Image.fromarray(uint8_img)
