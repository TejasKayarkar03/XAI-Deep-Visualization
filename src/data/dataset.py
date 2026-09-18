"""
Dataset management, preprocessing, and image transformation utilities.
Includes standard CIFAR-10 class labels, tensor transforms, base64 converters,
and curated sample generation for quick interactive exploration.
"""

import base64
import io
import os
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms, datasets

# Standard CIFAR-10 10-class labels
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Normalization statistics (ImageNet standard)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Preprocessing transforms
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

# For visualization: un-normalize back to [0, 1] RGB
unnormalize = transforms.Compose([
    transforms.Normalize(
        mean=[-m / s for m, s in zip(NORM_MEAN, NORM_STD)],
        std=[1.0 / s for s in NORM_STD]
    )
])


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Converts a normalized PyTorch tensor (C, H, W) to a PIL Image in RGB."""
    t = tensor.clone().detach().cpu()
    if t.ndim == 4:
        t = t.squeeze(0)
    t = unnormalize(t)
    t = torch.clamp(t, 0.0, 1.0)
    np_img = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(np_img)


def pil_to_tensor(image: Image.Image, image_size: Tuple[int, int] = (32, 32)) -> torch.Tensor:
    """Converts a PIL Image to a normalized PyTorch tensor (1, C, H, W)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])
    return transform(image).unsqueeze(0)


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Converts a PIL Image to a base64 encoded data URI."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def base64_to_pil(b64_string: str) -> Image.Image:
    """Converts a base64 string or data URI to a PIL Image."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    decoded = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(decoded)).convert("RGB")


def get_cifar10_loaders(data_dir: str = "./data", batch_size: int = 64, download: bool = True):
    """Returns train and test data loaders for CIFAR-10."""
    os.makedirs(data_dir, exist_ok=True)
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def generate_synthetic_demo_bank(output_dir: str = "./static/samples") -> List[dict]:
    """
    Creates a sample bank of reference images for the dashboard so testing can occur
    immediately without waiting for dataset downloads or manual uploads.
    """
    os.makedirs(output_dir, exist_ok=True)
    samples = []

    # Palette of distinctive colors and shapes for each category
    shapes = {
        "airplane": ("#38bdf8", "triangle"),
        "automobile": ("#ef4444", "car_box"),
        "bird": ("#fbbf24", "oval"),
        "cat": ("#a855f7", "cat_face"),
        "deer": ("#84cc16", "horns"),
        "dog": ("#f97316", "dog_face"),
        "frog": ("#10b981", "frog_eyes"),
        "horse": ("#d97706", "horse_silhouette"),
        "ship": ("#06b6d4", "boat_hull"),
        "truck": ("#6366f1", "truck_block")
    }

    for idx, name in enumerate(CLASS_NAMES):
        img_path = os.path.join(output_dir, f"{name}.png")
        if not os.path.exists(img_path):
            img = Image.new("RGB", (64, 64), color="#1e293b")
            draw = ImageDraw.Draw(img)
            color, shape = shapes.get(name, ("#ffffff", "oval"))

            # Draw distinct visual features for the model to detect
            if shape == "triangle":  # airplane
                draw.polygon([(32, 10), (14, 52), (50, 52)], fill=color)
                draw.polygon([(28, 25), (4, 40), (60, 40)], fill="#bae6fd")
            elif shape == "car_box":  # car
                draw.rectangle([(12, 30), (52, 50)], fill=color)
                draw.rectangle([(20, 18), (44, 30)], fill="#fca5a5")
                draw.ellipse([(16, 46), (26, 56)], fill="#0f172a")
                draw.ellipse([(38, 46), (48, 56)], fill="#0f172a")
            elif shape == "cat_face":  # cat
                draw.ellipse([(16, 20), (48, 52)], fill=color)
                draw.polygon([(16, 26), (22, 10), (30, 22)], fill="#e9d5ff")
                draw.polygon([(48, 26), (42, 10), (34, 22)], fill="#e9d5ff")
                draw.ellipse([(24, 32), (28, 36)], fill="#ffffff")
                draw.ellipse([(36, 32), (40, 36)], fill="#ffffff")
            elif shape == "boat_hull":  # ship
                draw.polygon([(10, 40), (54, 40), (46, 56), (18, 56)], fill=color)
                draw.rectangle([(30, 20), (34, 40)], fill="#ffffff")
                draw.polygon([(34, 22), (48, 30), (34, 38)], fill="#e0f2fe")
            elif shape == "truck_block":  # truck
                draw.rectangle([(10, 24), (40, 48)], fill=color)
                draw.rectangle([(40, 32), (54, 48)], fill="#a5b4fc")
                draw.ellipse([(16, 44), (24, 54)], fill="#0f172a")
                draw.ellipse([(42, 44), (50, 54)], fill="#0f172a")
            else:
                draw.ellipse([(16, 16), (48, 48)], fill=color)
                draw.rectangle([(26, 26), (38, 38)], fill="#f1f5f9")

            img.save(img_path)

        samples.append({
            "class_id": idx,
            "class_name": name,
            "filename": f"{name}.png",
            "url": f"/static/samples/{name}.png"
        })

    return samples
