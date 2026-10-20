"""Train a standard 2D U-Net baseline on the supplied HipMRI slice splits."""

import argparse
import json
import math
import platform
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DEFAULT_IMAGE_SIZE, HIPMRI_2D_ROOT, Nifti2DDataset, load_hipmri_splits
from modules import UNet2D
from utils import (DiceCELoss, SegmentationMetrics, count_parameters,
                   peak_vram_bytes, reset_peak_vram, synchronize)


CLASS_VALUES = (0, 1, 2, 3, 4, 5)


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=HIPMRI_2D_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/2d_unet"))
    parser.add_argument("--epochs", type=_positive_int, default=50)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--lr", type=_positive_float, default=1e-3)
    parser.add_argument("--base-channels", type=_positive_int, default=32)
    parser.add_argument("--image-size", type=_positive_int, nargs=2, default=DEFAULT_IMAGE_SIZE,
                        metavar=("HEIGHT", "WIDTH"), help="Shared image/mask size for batching.")
    parser.add_argument("--num-workers", type=_nonnegative_int, default=4)
    parser.add_argument("--seed", type=_nonnegative_int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-slices", type=_positive_int,
                        help="Use at most this many evenly spaced slices per split "
                             "for a smoke check, not a final experiment.")
    return parser.parse_args()


def run_epoch(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module,
              device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> dict:
    """Train or evaluate one split; metrics pool counts over the entire split."""
    training = optimizer is not None
    model.train(training)
    metrics = SegmentationMetrics(len(CLASS_VALUES))
    total_loss = 0.0
    samples = 0
    with torch.set_grad_enabled(training):
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered; inspect the data and learning rate.")
            if training:
                loss.backward()
                optimizer.step()
            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            samples += batch_size
            metrics.update(logits.argmax(dim=1), target)
    return {"loss": total_loss / samples, **metrics.compute()}


def write_json(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def plot_history(history: list[dict], path: Path) -> None:
    """Save training/validation loss, foreground Dice, and validation class Dice."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for split, label in (("train", "Train"), ("val", "Validation")):
        axes[0].plot(epochs, [row[split]["loss"] for row in history], label=label)
        axes[1].plot(epochs, [row[split]["mean_foreground_dice"] for row in history],
                     label=label)
    for channel, label in enumerate(CLASS_VALUES):
        axes[2].plot(epochs, [row["val"]["dice"][channel] for row in history],
                     label=f"Label {label}" + (" (background)" if channel == 0 else ""))
    axes[0].set_title("Dice + cross-entropy loss")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Mean foreground Dice")
    axes[2].set_title("Validation Dice per class")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[1:]:
        ax.set_ylabel("Dice")
        ax.set_ylim(0, 1)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _format_score(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    random.seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    pairs = load_hipmri_splits(args.data_root)
    full_split_sizes = {split: len(items) for split, items in pairs.items()}
    if args.max_slices is not None:
        for split, items in pairs.items():
            if args.max_slices < len(items):
                indices = torch.linspace(0, len(items) - 1, args.max_slices).long().tolist()
                pairs[split] = [items[index] for index in indices]
        print("SMOKE CHECK: --max-slices limits the experiment; do not report as final results.")
    datasets = {split: Nifti2DDataset(items, class_values=CLASS_VALUES,
                                    image_size=tuple(args.image_size))
                for split, items in pairs.items()}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train",
                          num_workers=args.num_workers, pin_memory=device.type == "cuda",
                          generator=generator if split == "train" else None)
        for split, dataset in datasets.items()
    }
    model_config = {"in_channels": 1, "num_classes": len(CLASS_VALUES),
                    "base_channels": args.base_channels}
    model = UNet2D(**model_config).to(device)
    criterion = DiceCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    parameter_count = count_parameters(model)
    hardware = (torch.cuda.get_device_name(device) if device.type == "cuda"
                else platform.processor() or platform.machine())
    print(f"Device: {device} ({hardware}); parameters: {parameter_count:,}")
    print("Slices: " + ", ".join(f"{split}={len(data)}" for split, data in datasets.items()))
    print("Metrics pool pixel counts across slices; foreground mean excludes background "
          "and classes absent from both prediction and truth.")

    best_score = None
    best_epoch = None
    best_path = args.output_dir / "best.pt"
    history = []
    reset_peak_vram(device)
    synchronize(device)
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_scores = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_scores = run_epoch(model, loaders["val"], criterion, device)
        history.append({"epoch": epoch, "train": train_scores, "val": val_scores})
        score = val_scores["mean_foreground_dice"]
        if best_epoch is None or (score is not None and (best_score is None or score > best_score)):
            best_score = score
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_config": model_config,
                        "class_values": list(CLASS_VALUES), "image_size": list(args.image_size),
                        "epoch": epoch,
                        "best_val_mean_foreground_dice": best_score, "args": config}, best_path)
        write_json(args.output_dir / "history.json", history)
        print(f"Epoch {epoch:03d}/{args.epochs}: train loss={train_scores['loss']:.4f}, "
              f"val loss={val_scores['loss']:.4f}, "
              f"val foreground Dice={_format_score(score)}")
        print("  Val Dice " + ", ".join(
            f"label {label}={_format_score(value)}"
            for label, value in zip(CLASS_VALUES, val_scores["dice"])))
    synchronize(device)
    training_seconds = perf_counter() - start
    training_peak_vram = peak_vram_bytes(device)
    print(f"Best checkpoint: {best_path} (epoch {best_epoch}, "
          f"val foreground Dice={_format_score(best_score)})")
    print(f"Training/validation time: {training_seconds:.2f} s; "
          f"peak allocated CUDA bytes: {training_peak_vram}")

    # The test split is evaluated once, only after validation chose the checkpoint.
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    test_scores = run_epoch(model, loaders["test"], criterion, device)
    result = {
        "model_config": model_config, "class_values": list(CLASS_VALUES),
        "image_size": list(args.image_size), "args": config,
        "seed": args.seed, "device": str(device), "hardware": hardware,
        "torch_version": str(torch.__version__), "parameter_count": parameter_count,
        "split_sizes": {split: len(data) for split, data in datasets.items()},
        "full_split_sizes": full_split_sizes, "max_slices": args.max_slices,
        "subset_evaluation": any(len(pairs[split]) < full_split_sizes[split] for split in pairs),
        "best_checkpoint": str(best_path), "best_epoch": best_epoch,
        "best_val_mean_foreground_dice": best_score, "history": history,
        "test": test_scores, "training_seconds": training_seconds,
        "training_time_scope": "Training and validation loop, including checkpoint/history "
                               "writes; excludes setup, final test evaluation and plotting.",
        "peak_training_vram_bytes": training_peak_vram,
        "peak_vram_scope": "Peak PyTorch CUDA tensor allocation during training/validation; "
                           "includes model and optimizer, excludes reserved unused memory.",
        "metric_aggregation": "Pixel counts pooled over each split's slices. Per-class Dice "
                              "and IoU include background; mean foreground Dice excludes "
                              "background and classes absent from prediction and truth. "
                              "Loss is the sample-weighted mean of batch losses.",
    }
    write_json(args.output_dir / "results.json", result)
    plot_history(history, args.output_dir / "curves.png")
    print("Test Dice " + ", ".join(
        f"label {label}={_format_score(value)}"
        for label, value in zip(CLASS_VALUES, test_scores["dice"])))
    print(f"Test foreground Dice: {_format_score(test_scores['mean_foreground_dice'])}")
    print(f"Saved history.json, results.json and curves.png in {args.output_dir}")


if __name__ == "__main__":
    main()
