"""
Activation Maximization (Feature Dreaming & Synthesis).
Synthesizes synthetic images from random Gaussian noise using gradient ascent
to reveal the visual pattern that maximally excites a chosen filter, channel, or class.
Reference: Erhan et al. (2009), Olah et al. (Distill, 2017).
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from src.data.dataset import unnormalize


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Computes total variation loss to enforce spatial smoothness."""
    diff_h = img[:, :, 1:, :] - img[:, :, :-1, :]
    diff_w = img[:, :, :, 1:] - img[:, :, :, :-1]
    return torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_w))


def random_jitter(img: torch.Tensor, max_jitter: int = 2) -> Tuple[torch.Tensor, int, int]:
    """Applies random 2D spatial jitter to prevent pixel artifacts."""
    ox = int(np.random.randint(-max_jitter, max_jitter + 1))
    oy = int(np.random.randint(-max_jitter, max_jitter + 1))
    return torch.roll(img, shifts=(ox, oy), dims=(2, 3)), ox, oy


def unjitter(img: torch.Tensor, ox: int, oy: int) -> torch.Tensor:
    """Reverses spatial jitter."""
    return torch.roll(img, shifts=(-ox, -oy), dims=(2, 3))


def maximize_feature_activation(
    model: nn.Module,
    layer_name: Optional[str] = None,
    channel_idx: int = 0,
    target_class: Optional[int] = None,
    steps: int = 60,
    lr: float = 0.1,
    tv_weight: float = 0.01,
    l2_weight: float = 0.005,
    image_size: Tuple[int, int] = (64, 64)
) -> Image.Image:
    """
    Executes gradient ascent on an initially random input image to maximize
    activation of a specific internal channel or final class score.
    """
    model.eval()
    device = next(model.parameters()).device

    # Initialize random Gaussian noise image centered at 0.5
    input_tensor = (torch.randn(1, 3, image_size[0], image_size[1], device=device) * 0.1 + 0.5).requires_grad_(True)
    optimizer = torch.optim.Adam([input_tensor], lr=lr, weight_decay=l2_weight)

    activation_holder = {}
    hook_handle = None

    if layer_name is not None:
        if hasattr(model, "get_layer_by_name"):
            target_module = model.get_layer_by_name(layer_name)
        else:
            target_module = dict(model.named_modules()).get(layer_name)

        if target_module is not None:
            def hook(module, inp, out):
                activation_holder["act"] = out
            hook_handle = target_module.register_forward_hook(hook)

    try:
        for step in range(steps):
            optimizer.zero_grad()

            # Apply random translation jitter
            jittered, ox, oy = random_jitter(input_tensor, max_jitter=2)

            logits = model(jittered)

            if layer_name is not None and "act" in activation_holder:
                act = activation_holder["act"]
                ch = min(channel_idx, act.shape[1] - 1)
                # Maximize mean activation of the designated channel
                loss = -torch.mean(act[:, ch])
            elif target_class is not None:
                # Maximize score for target class
                loss = -logits[0, target_class]
            else:
                # Default: maximize max logit
                loss = -torch.max(logits)

            # Regularization penalty
            tv_loss = total_variation_loss(jittered) * tv_weight
            total_loss = loss + tv_loss

            total_loss.backward()
            optimizer.step()

            # Reverse jitter and clamp to valid range
            with torch.no_grad():
                input_tensor.clamp_(0.0, 1.0)

    finally:
        if hook_handle is not None:
            hook_handle.remove()

    # Convert final optimized tensor to PIL Image
    img_np = input_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    img_min, img_max = img_np.min(), img_np.max()
    if img_max > img_min:
        img_np = (img_np - img_min) / (img_max - img_min)
    else:
        img_np = np.zeros_like(img_np)

    final_uint8 = (img_np * 255.0).astype(np.uint8)
    return Image.fromarray(final_uint8)
