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
from src.data.dataset import get_cifar10_loaders, CLASS_NAMES


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
    epochs: int = 5,
    batch_size: int = 64,
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
        print("[Training] Synthesizing structured visual patterns for feature learning...")
        num_samples = 1500
        x_data = torch.randn(num_samples, 3, 32, 32) * 0.1
        y_data = torch.randint(0, len(CLASS_NAMES), (num_samples,))

        # Inject distinctive visual features per category:
        # Edge lines, circular spots, diagonal textures, cross-hairs
        for i in range(num_samples):
            c = y_data[i].item()
            grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij")
            if c == 0:  # Airplane - horizontal wings and vertical body
                x_data[i, :, 14:18, :] += 0.9
                x_data[i, :, :, 14:18] += 0.9
            elif c == 1:  # Automobile - lower box and wheels
                x_data[i, 0, 16:26, 4:28] += 1.0
                x_data[i, :, 24:28, 6:12] -= 0.8
                x_data[i, :, 24:28, 20:26] -= 0.8
            elif c == 2:  # Bird - diagonal streaks
                diag = (grid_x + grid_y).abs() < 0.2
                x_data[i, 1] += diag.float() * 1.2
            elif c == 3:  # Cat - concentric circles / eyes
                dist = torch.sqrt(grid_x**2 + grid_y**2)
                x_data[i, 2] += (dist < 0.5).float() * 1.2
            elif c == 4:  # Deer - branching vertical lines
                x_data[i, 1, :, 8:12] += 0.8
                x_data[i, 1, :, 20:24] += 0.8
            elif c == 5:  # Dog - center blob with high frequency noise
                x_data[i, 0] += torch.exp(-(grid_x**2 + grid_y**2) / 0.3) * 1.5
            elif c == 6:  # Frog - green dominant circular texture
                x_data[i, 1] += (torch.sin(grid_x * 8) * torch.cos(grid_y * 8) > 0.3).float() * 1.4
            elif c == 7:  # Horse - elongated oval
                x_data[i, 0] += ((grid_x**2 / 0.6 + grid_y**2 / 0.2) < 0.8).float() * 1.1
            elif c == 8:  # Ship - lower horizontal block with cyan tint
                x_data[i, 1:, 18:28, :] += 1.0
            elif c == 9:  # Truck - dense grid pattern
                grid = ((torch.sin(grid_x * 12) > 0) & (torch.cos(grid_y * 12) > 0)).float()
                x_data[i, :] += grid * 1.0

        val_split = int(num_samples * 0.8)
        train_ds = TensorDataset(x_data[:val_split], y_data[:val_split])
        test_ds = TensorDataset(x_data[val_split:], y_data[val_split:])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

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

