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


def draw_category_vector(name: str) -> Image.Image:
    """Renders a high-contrast, visually distinctive archetype for a CIFAR-10 category."""
    img = Image.new("RGB", (64, 64), color="#1e293b")
    draw = ImageDraw.Draw(img)

    if name == "airplane":
        draw.polygon([(32, 6), (28, 52), (36, 52)], fill="#38bdf8")
        draw.polygon([(32, 24), (6, 40), (58, 40)], fill="#0284c7")
        draw.polygon([(32, 48), (18, 58), (46, 58)], fill="#0369a1")
        draw.ellipse([(30, 12), (34, 20)], fill="#e0f2fe")
    elif name == "automobile":
        draw.rectangle([(10, 32), (54, 48)], fill="#ef4444")
        draw.polygon([(18, 32), (24, 20), (40, 20), (46, 32)], fill="#f87171")
        draw.polygon([(24, 22), (38, 22), (43, 30), (20, 30)], fill="#93c5fd")
        draw.ellipse([(14, 44), (24, 54)], fill="#0f172a")
        draw.ellipse([(40, 44), (50, 54)], fill="#0f172a")
        draw.ellipse([(17, 47), (21, 51)], fill="#e2e8f0")
        draw.ellipse([(43, 47), (47, 51)], fill="#e2e8f0")
        draw.rectangle([(50, 36), (54, 40)], fill="#fef08a")
    elif name == "bird":
        draw.ellipse([(18, 24), (46, 48)], fill="#eab308")
        draw.ellipse([(36, 16), (52, 32)], fill="#facc15")
        draw.polygon([(48, 22), (58, 26), (48, 30)], fill="#f97316")
        draw.ellipse([(42, 20), (46, 24)], fill="#0f172a")
        draw.polygon([(20, 28), (34, 28), (14, 42)], fill="#ca8a04")
        draw.polygon([(18, 36), (6, 44), (16, 48)], fill="#ca8a04")
    elif name == "cat":
        draw.ellipse([(16, 22), (48, 52)], fill="#a855f7")
        draw.polygon([(16, 28), (22, 10), (30, 24)], fill="#c084fc")
        draw.polygon([(48, 28), (42, 10), (34, 24)], fill="#c084fc")
        draw.ellipse([(22, 30), (28, 38)], fill="#fef08a")
        draw.ellipse([(36, 30), (42, 38)], fill="#fef08a")
        draw.ellipse([(24, 32), (26, 36)], fill="#0f172a")
        draw.ellipse([(38, 32), (40, 36)], fill="#0f172a")
        draw.polygon([(30, 40), (34, 40), (32, 43)], fill="#f43f5e")
    elif name == "deer":
        draw.polygon([(24, 44), (40, 44), (36, 60), (28, 60)], fill="#92400e")
        draw.ellipse([(22, 26), (42, 50)], fill="#b45309")
        draw.line([(26, 26), (14, 10)], fill="#fef3c7", width=2)
        draw.line([(20, 18), (12, 20)], fill="#fef3c7", width=2)
        draw.line([(17, 14), (21, 9)], fill="#fef3c7", width=2)
        draw.line([(38, 26), (50, 10)], fill="#fef3c7", width=2)
        draw.line([(44, 18), (52, 20)], fill="#fef3c7", width=2)
        draw.line([(47, 14), (43, 9)], fill="#fef3c7", width=2)
        draw.ellipse([(15, 25), (23, 31)], fill="#d97706")
        draw.ellipse([(41, 25), (49, 31)], fill="#d97706")
        draw.ellipse([(26, 34), (30, 38)], fill="#000000")
        draw.ellipse([(34, 34), (38, 38)], fill="#000000")
    elif name == "dog":
        draw.ellipse([(18, 20), (46, 48)], fill="#ea580c")
        draw.ellipse([(12, 24), (22, 46)], fill="#c2410c")
        draw.ellipse([(42, 24), (52, 46)], fill="#c2410c")
        draw.ellipse([(24, 34), (40, 48)], fill="#ffedd5")
        draw.ellipse([(29, 36), (35, 42)], fill="#18181b")
        draw.ellipse([(24, 26), (28, 30)], fill="#18181b")
        draw.ellipse([(36, 26), (40, 30)], fill="#18181b")
    elif name == "frog":
        draw.ellipse([(14, 24), (50, 52)], fill="#10b981")
        draw.ellipse([(16, 12), (28, 26)], fill="#059669")
        draw.ellipse([(36, 12), (48, 26)], fill="#059669")
        draw.ellipse([(19, 15), (25, 23)], fill="#ffffff")
        draw.ellipse([(39, 15), (45, 23)], fill="#ffffff")
        draw.ellipse([(21, 17), (24, 21)], fill="#000000")
        draw.ellipse([(41, 17), (44, 21)], fill="#000000")
        draw.arc([(22, 34), (42, 44)], start=0, end=180, fill="#047857", width=2)
    elif name == "horse":
        draw.polygon([(18, 34), (32, 36), (30, 60), (14, 60)], fill="#78350f")
        draw.polygon([(20, 20), (32, 16), (44, 32), (38, 48), (24, 38)], fill="#b45309")
        draw.ellipse([(34, 40), (46, 52)], fill="#92400e")
        draw.polygon([(28, 16), (32, 8), (36, 16)], fill="#d97706")
        draw.line([(22, 18), (14, 26)], fill="#451a03", width=3)
        draw.line([(20, 24), (12, 32)], fill="#451a03", width=3)
        draw.line([(18, 30), (10, 38)], fill="#451a03", width=3)
        draw.ellipse([(28, 24), (32, 28)], fill="#000000")
    elif name == "ship":
        draw.rectangle([(0, 48), (64, 64)], fill="#0284c7")
        draw.polygon([(10, 42), (54, 42), (46, 54), (18, 54)], fill="#f8fafc")
        draw.line([(32, 14), (32, 42)], fill="#78350f", width=2)
        draw.polygon([(34, 16), (50, 36), (34, 36)], fill="#06b6d4")
        draw.polygon([(30, 20), (18, 36), (30, 36)], fill="#38bdf8")
    elif name == "truck":
        draw.rectangle([(8, 18), (38, 48)], fill="#6366f1")
        draw.rectangle([(38, 26), (56, 48)], fill="#4f46e5")
        draw.rectangle([(46, 28), (54, 36)], fill="#c7d2fe")
        draw.ellipse([(44, 44), (54, 54)], fill="#0f172a")
        draw.ellipse([(12, 44), (22, 54)], fill="#0f172a")
        draw.ellipse([(24, 44), (34, 54)], fill="#0f172a")
        draw.ellipse([(47, 47), (51, 51)], fill="#e2e8f0")
        draw.ellipse([(15, 47), (19, 51)], fill="#e2e8f0")
        draw.ellipse([(27, 47), (31, 51)], fill="#e2e8f0")

    return img


def generate_synthetic_demo_bank(output_dir: str = "./static/samples", overwrite: bool = True) -> List[dict]:
    """
    Creates a sample bank of reference images for the dashboard so testing can occur
    immediately without waiting for dataset downloads or manual uploads.
    """
    os.makedirs(output_dir, exist_ok=True)
    samples = []

    for idx, name in enumerate(CLASS_NAMES):
        img_path = os.path.join(output_dir, f"{name}.png")
        if overwrite or not os.path.exists(img_path):
            img = draw_category_vector(name)
            img.save(img_path)

        samples.append({
            "class_id": idx,
            "class_name": name,
            "filename": f"{name}.png",
            "url": f"/static/samples/{name}.png"
        })

    return samples


def get_synthetic_augmented_loaders(batch_size: int = 32, num_reps: int = 80):
    """
    Synthesizes a robust augmented dataset matching the exact preprocessing pipeline
    so the model learns rich, distinct hierarchical features for all 10 categories.
    """
    base_images = [draw_category_vector(c) for c in CLASS_NAMES]

    aug_train_tx = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    aug_test_tx = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    train_x, train_y = [], []
    for _ in range(num_reps):
        for idx in range(len(CLASS_NAMES)):
            train_x.append(aug_train_tx(base_images[idx]))
            train_y.append(idx)

    test_x, test_y = [], []
    for _ in range(16):
        for idx in range(len(CLASS_NAMES)):
            test_x.append(aug_train_tx(base_images[idx]))
            test_y.append(idx)

    train_ds = torch.utils.data.TensorDataset(torch.stack(train_x), torch.tensor(train_y))
    test_ds = torch.utils.data.TensorDataset(torch.stack(test_x), torch.tensor(test_y))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

