"""
Training and Evaluation Pipeline for DeepVisualNet.
Trains on CIFAR-10 / benchmark visual data, tracks metrics,
and saves model checkpoints for XAI visualization.
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.cnn_model import DeepVisualNet
from src.data.dataset import get_cifar10_loaders, get_synthetic_augmented_loaders, CLASS_NAMES


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device):
    """Executes a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = 100.0 * correct / max(total, 1)
    return epoch_loss, epoch_acc


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    """Evaluates the model on test/validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / max(total, 1)
    val_acc = 100.0 * correct / max(total, 1)
    return val_loss, val_acc


def train_model(
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.001,
    dataset_name: str = "synthetic",
    save_path: str = "models/checkpoints/deep_visual_net.pth"
):
    """Main training orchestration function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using compute device: {device}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model = DeepVisualNet(num_classes=len(CLASS_NAMES)).to(device)

    train_loader = None
    test_loader = None

    if dataset_name.lower() == "cifar10":
        try:
            print("[Training] Initializing CIFAR-10 dataset...")
            train_loader, test_loader = get_cifar10_loaders(data_dir="./data", batch_size=batch_size, download=True)
            print("[Training] CIFAR-10 loaded successfully.")
        except Exception as e:
            print(f"[Training] Warning: Could not download CIFAR-10 ({e}). Falling back to synthetic visual patterns.")
            dataset_name = "synthetic"

    if dataset_name.lower() == "synthetic":
        print("[Training] Generating augmented visual pattern dataset...")
        train_loader, test_loader = get_synthetic_augmented_loaders(batch_size=batch_size, num_reps=80)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch [{epoch}/{epochs}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "class_names": CLASS_NAMES
            }, save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Val Acc: {val_acc:.2f}%)")

    total_time = time.time() - start_time
    print(f"\n[Training Complete] Total time: {total_time:.1f}s | Best Val Acc: {best_acc:.2f}%")
    return model, best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepVisualNet for Feature Visualization")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "cifar10"], help="Dataset to train on")
    parser.add_argument("--save-path", type=str, default="models/checkpoints/deep_visual_net.pth", help="Checkpoint path")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, dataset_name=args.dataset, save_path=args.save_path)

