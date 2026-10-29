"""Train the 2D U-Net baseline or the 3D Improved U-Net on the same HipMRI patient split.

    python train.py --model unet2d       # 2D U-Net on axial slices
    python train.py --model improved3d   # 3D Improved U-Net on whole scans

Both models read the same 2×-downsampled scans (--downsample 1 keeps full resolution)
and patient split from dataset.py, with the same training augmentation (--no-augment
turns it off). The first run creates patient_split.json if it is missing. Both use the same Dice + cross-entropy
loss and Adam schedule, and they are validated identically: each
epoch, whole validation scans are predicted (slice by slice for the 2D model) and
scored on the working grid, and the checkpoint with the best mean foreground Dice over
validation patients is kept. That checkpoint is tested once on the test patients
against the full-resolution labels (utils.evaluate_scans). Outputs in runs/<model>/:
best.pt, history.json, results.json and curves.png.
"""

import argparse
import json
import math
import platform
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (DEFAULT_DOWNSAMPLE, DEFAULT_ROOT, IMAGE_DIR, LABEL_DIR, LABEL_NAMES,
                     SPLIT_FILE, Augmentation, AxialSliceDataset, Scan, VolumeDataset,
                     create_split, discover_scans, load_scan, load_split, split_scans)
from utils import (CHART_RC, INK_MUTED, LABEL_COLORS, MODELS, DiceCELoss, ModelSpec,
                   SegmentationMetrics, confusion_matrix, count_parameters, evaluate_scans,
                   format_summary, patient_summary, peak_vram_bytes, predict_volume,
                   reset_peak_vram, scores_from_confusion, summarize, synchronize)


# Per-model defaults; each can be overridden on the command line.
MODEL_DEFAULTS = {
    "unet2d": {"base_channels": 32, "batch_size": 16, "lr": 1e-3},
    "improved3d": {"base_channels": 16, "batch_size": 1, "lr": 5e-4},
}
PROSTATE = LABEL_NAMES.index("prostate")


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


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive finite number")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--label-dir", default=LABEL_DIR)
    parser.add_argument("--split-file", type=Path, default=SPLIT_FILE)
    parser.add_argument("--downsample", type=_positive_int, default=DEFAULT_DOWNSAMPLE,
                        help="Integer factor for every axis (default 2); 1 keeps full resolution.")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                        help="Random flip, rotation, scaling and intensity changes on the "
                             "training data (default: on).")
    parser.add_argument("--output-dir", type=Path, help="Default: runs/<model>.")
    parser.add_argument("--epochs", type=_positive_int, default=50)
    parser.add_argument("--batch-size", type=_positive_int,
                        help="Default: 16 axial slices (2D) or 1 scan (3D).")
    parser.add_argument("--lr", type=_positive_float, help="Default: 1e-3 (2D) or 5e-4 (3D).")
    parser.add_argument("--lr-decay", type=_positive_float, default=0.985,
                        help="Learning-rate factor applied after every epoch.")
    parser.add_argument("--weight-decay", type=_nonnegative_float, default=1e-5)
    parser.add_argument("--base-channels", type=_positive_int, help="Default: 32 (2D) or 16 (3D).")
    parser.add_argument("--num-workers", type=_nonnegative_int, default=4)
    parser.add_argument("--seed", type=_nonnegative_int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-scans", type=_positive_int,
                        help="Use at most this many evenly spaced scans per split "
                             "for a smoke check, not a final experiment.")
    args = parser.parse_args()
    for key, value in MODEL_DEFAULTS[args.model].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.model == "improved3d" and args.batch_size != 1:
        parser.error("The 3D model trains on whole scans, which can differ in size; "
                     "use --batch-size 1.")
    if args.output_dir is None:
        args.output_dir = Path("runs") / args.model
    return args


def train_epoch(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device) -> dict:
    """One pass over the training data; Dice pools every training voxel seen."""
    model.train()
    metrics = SegmentationMetrics(len(LABEL_NAMES))
    total_loss = 0.0
    samples = 0
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss encountered; inspect the data and learning rate.")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.shape[0]
        samples += images.shape[0]
        metrics.update(logits.detach().argmax(dim=1), target)
    return {"loss": total_loss / samples, **metrics.compute()}


def validate(model: torch.nn.Module, volumes: list[tuple[Scan, torch.Tensor, torch.Tensor]],
             criterion: torch.nn.Module, spec: ModelSpec, device: torch.device) -> dict:
    """Score whole validation scans on the working grid; means are over patients."""
    model.eval()
    losses, records = [], []
    for _, image, target in volumes:
        logits = predict_volume(model, image.to(device), slicewise=spec.slicewise)
        target = target.to(device).long()
        losses.append(criterion(logits, target.unsqueeze(0)).item())
        confusion = confusion_matrix(logits.argmax(dim=1)[0], target, len(LABEL_NAMES))
        records.append(scores_from_confusion(confusion))
    patients = [scan.patient_id for scan, _, _ in volumes]

    def patient_mean(values: list) -> float | None:
        return patient_summary(values, patients)["mean"]

    return {
        "loss": mean(losses),
        "dice": [patient_mean([r["dice"][c] for r in records]) for c in range(len(LABEL_NAMES))],
        "iou": [patient_mean([r["iou"][c] for r in records]) for c in range(len(LABEL_NAMES))],
        "mean_foreground_dice": patient_mean([r["mean_foreground_dice"] for r in records]),
        "mean_foreground_iou": patient_mean([r["mean_foreground_iou"] for r in records]),
    }


def write_json(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def plot_history(history: list[dict], spec: ModelSpec, path: Path) -> None:
    """Save loss, mean foreground Dice and per-class validation Dice per epoch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = [row["epoch"] for row in history]
    with plt.rc_context(CHART_RC):
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), layout="constrained")
        for split, label, color in (("train", "Training", INK_MUTED),
                                    ("val", "Validation", spec.color)):
            axes[0].plot(epochs, [row[split]["loss"] for row in history], color=color, label=label)
            axes[1].plot(epochs, [row[split]["mean_foreground_dice"] for row in history],
                         color=color, label=label)
        for channel, name in enumerate(LABEL_NAMES[1:], start=1):
            axes[2].plot(epochs, [row["val"]["dice"][channel] for row in history],
                         color=LABEL_COLORS[name], label=name)
        axes[0].set(title="Dice + cross-entropy loss", ylabel="Loss")
        axes[1].set(title="Mean foreground Dice", ylabel="Dice", ylim=(0, 1))
        axes[2].set(title="Validation Dice per class", ylabel="Dice", ylim=(0, 1))
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(loc="best")
        fig.suptitle(f"{spec.label}: training is pooled over batches; "
                     "validation is the mean over patients of whole-scan scores")
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
    spec = MODELS[args.model]

    all_scans = discover_scans(args.data_root, args.image_dir, args.label_dir)
    if not args.split_file.exists():
        try:
            create_split(all_scans, args.split_file)
            print(f"Created the patient split {args.split_file}; commit it so that every run "
                  "and predict.py use the same patients.")
        except FileExistsError:
            pass  # A run started at the same time wrote it: same data and seed, same split.
    split = load_split(args.split_file)
    scans = split_scans(all_scans, split)
    full_scan_counts = {name: len(items) for name, items in scans.items()}
    if args.max_scans is not None:
        for name, items in scans.items():
            if args.max_scans < len(items):
                indices = torch.linspace(0, len(items) - 1, args.max_scans).long().tolist()
                scans[name] = [items[index] for index in indices]
        print("SMOKE CHECK: --max-scans limits the experiment; do not report these results.")
    print("Scans: " + ", ".join(f"{name}={len(items)}" for name, items in scans.items())
          + f"; downsampled by {args.downsample}; augmentation {'on' if args.augment else 'off'}. "
          "Loading training and validation scans...")
    augmentation = Augmentation() if args.augment else None
    dataset_class = AxialSliceDataset if spec.slicewise else VolumeDataset
    train_data = dataset_class(scans["train"], downsample=args.downsample,
                               augmentation=augmentation)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(args.seed),
                        persistent_workers=args.num_workers > 0)
    val_volumes = []
    for scan in scans["val"]:
        image, target, _ = load_scan(scan, args.downsample)
        # uint8 labels keep the cached validation set small at full resolution.
        val_volumes.append((scan, image, target.to(torch.uint8)))

    model_config = {"in_channels": 1, "num_classes": len(LABEL_NAMES),
                    "base_channels": args.base_channels}
    model = spec.cls(**model_config).to(device)
    criterion = DiceCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    parameter_count = count_parameters(model)
    hardware = (torch.cuda.get_device_name(device) if device.type == "cuda"
                else platform.processor() or platform.machine())
    print(f"{spec.label} on {device} ({hardware}); parameters: {parameter_count:,}; "
          f"training samples: {len(train_data)} {'axial slices' if spec.slicewise else 'scans'}")

    best_score = None
    best_epoch = None
    best_path = args.output_dir / "best.pt"
    history = []
    reset_peak_vram(device)
    synchronize(device)
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_scores = train_epoch(model, loader, criterion, optimizer, device)
        val_scores = validate(model, val_volumes, criterion, spec, device)
        scheduler.step()
        history.append({"epoch": epoch, "lr": learning_rate,
                        "seconds": perf_counter() - epoch_start,
                        "train": train_scores, "val": val_scores})
        score = val_scores["mean_foreground_dice"]
        if best_epoch is None or (score is not None and (best_score is None or score > best_score)):
            best_score = score
            best_epoch = epoch
            torch.save({"model": args.model, "model_state_dict": model.state_dict(),
                        "model_config": model_config, "label_names": list(LABEL_NAMES),
                        "downsample": args.downsample, "split": split, "epoch": epoch,
                        "best_val_mean_foreground_dice": best_score, "args": config}, best_path)
        write_json(args.output_dir / "history.json", history)
        print(f"Epoch {epoch:03d}/{args.epochs}: train loss={train_scores['loss']:.4f}, "
              f"val loss={val_scores['loss']:.4f}, val foreground Dice={_format_score(score)}, "
              f"val prostate Dice={_format_score(val_scores['dice'][PROSTATE])}, "
              f"{history[-1]['seconds']:.0f} s")
    synchronize(device)
    training_seconds = perf_counter() - start
    training_peak_vram = peak_vram_bytes(device)
    print(f"Best checkpoint: {best_path} (epoch {best_epoch}, "
          f"val foreground Dice={_format_score(best_score)})")

    # The test patients are scored once, after validation chose the checkpoint.
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    records = evaluate_scans(model, scans["test"], slicewise=spec.slicewise,
                             downsample=args.downsample, device=device, organ=PROSTATE)
    test = summarize(records)
    result = {
        "model": args.model, "model_config": model_config, "label_names": list(LABEL_NAMES),
        "args": config, "augmentation": asdict(augmentation) if augmentation else None,
        "device": str(device), "hardware": hardware,
        "torch_version": str(torch.__version__), "parameter_count": parameter_count,
        "split_patients": {name: len(patients) for name, patients in split.items()},
        "scans": {name: len(items) for name, items in scans.items()},
        "full_scan_counts": full_scan_counts,
        "subset_evaluation": any(len(scans[name]) < full_scan_counts[name] for name in scans),
        "best_checkpoint": str(best_path), "best_epoch": best_epoch,
        "best_val_mean_foreground_dice": best_score,
        "training_seconds": training_seconds, "peak_training_vram_bytes": training_peak_vram,
        "scope": {
            "validation": "Whole scans on the working (downsampled) grid; per-scan scores "
                          "averaged per patient, then over patients.",
            "test": "Whole scans scored against the full-resolution labels; summaries give "
                    "the mean and SD over patients and pooled scores (see utils.py).",
            "training_time": "Training and validation loop including checkpoint writes; "
                             "excludes data loading before the first epoch and testing.",
            "peak_vram": "Peak PyTorch CUDA tensor allocation during training and validation.",
        },
        "history": history, "test": test, "test_scans": records,
    }
    write_json(args.output_dir / "results.json", result)
    plot_history(history, spec, args.output_dir / "curves.png")
    print(f"Training/validation time: {training_seconds:.1f} s; "
          f"peak allocated CUDA bytes: {training_peak_vram}")
    print(f"Test ({test['patients']} patients, {test['scans']} scans, full-resolution labels; "
          "mean ± SD over patients):")
    for channel, name in enumerate(LABEL_NAMES[1:], start=1):
        print(f"  {name:<10} Dice {format_summary(test['dice'][channel]):<15} "
              f"IoU {format_summary(test['iou'][channel])}")
    print(f"  foreground Dice {format_summary(test['mean_foreground_dice'])}, "
          f"IoU {format_summary(test['mean_foreground_iou'])}")
    print("  prostate Dice by third: " + ", ".join(
        f"{name} {format_summary(summary)}"
        for name, summary in zip(("apex", "mid", "base"), test["organ_thirds_dice"])))
    print(f"Saved best.pt, history.json, results.json and curves.png in {args.output_dir}")


if __name__ == "__main__":
    main()
