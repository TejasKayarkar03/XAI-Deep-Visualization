"""
Grad-CAM: Gradient-weighted Class Activation Mapping.
Computes spatial attribution heatmaps for target classes by weighting
activation maps with their backpropagated gradients.
Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
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
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM interpreter for convolutional neural networks.
    """
    def __init__(self, model: nn.Module, target_layer_name: str = "conv4"):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        if hasattr(self.model, "get_layer_by_name"):
            layer = self.model.get_layer_by_name(self.target_layer_name)
        else:
            layer = dict(self.model.named_modules()).get(self.target_layer_name)

        if layer is None:
            raise ValueError(f"Layer {self.target_layer_name} not found.")

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hooks.append(layer.register_forward_hook(forward_hook))
        self.hooks.append(layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates 2D normalized Grad-CAM heatmap for the specified class (or predicted class).
        Returns:
            heatmap: 2D numpy array in [0, 1] with shape matching input_tensor spatial dims.
            target_class: the class evaluated.
            confidence: softmax probability for target class.
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device).requires_grad_(True)

        self.model.zero_grad()
        logits = self.model(input_tensor)
        probabilities = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        confidence = float(probabilities[0, target_class].item())

        # Backward pass for the target class score
        score.backward(retain_graph=True)

        # Activations and Gradients shape: (1, C, H, W)
        grads = self.gradients.detach()
        acts = self.activations.detach()

        # Global average pooling of gradients: weights alpha_k
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Linear combination of maps weighted by alpha
        cam = torch.sum(weights * acts, dim=1, keepdim=True)   # (1, 1, H, W)

        # Apply ReLU to retain only positive influences on the class
        cam = F.relu(cam)

        # Upsample to match input resolution
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            heatmap = (cam - c_min) / (c_max - c_min)
        else:
            heatmap = np.zeros_like(cam)

        return heatmap, target_class, confidence


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    colormap_name: str = "jet"
) -> Image.Image:
    """
    Overlays a Grad-CAM heatmap over the original PIL Image using alpha blending.
    """
    orig_resized = original_image.resize((heatmap.shape[1], heatmap.shape[0]), Image.Resampling.BILINEAR)
    orig_np = np.array(orig_resized, dtype=np.float32) / 255.0

    cmap = matplotlib.colormaps.get_cmap(colormap_name)
    colored_cam = cmap(heatmap)[:, :, :3]  # Drop alpha channel

    # Alpha composite
    blended = (1.0 - alpha) * orig_np + alpha * colored_cam
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)
