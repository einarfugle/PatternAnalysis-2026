"""Evaluate a trained 2D U-Net and save slice-level comparison figures."""

import argparse
import csv
import json
import platform
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import torch

from dataset import HIPMRI_2D_ROOT, Nifti2DDataset, load_hipmri_splits
from modules import UNet2D
from utils import (SegmentationMetrics, count_parameters, peak_vram_bytes,
                   reset_peak_vram, timed_inference)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=HIPMRI_2D_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("predictions/2d_unet"))
    parser.add_argument("--num-figures", type=_nonnegative_int, default=5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-slices", type=_positive_int,
                        help="Evaluate an evenly spaced subset for a smoke check.")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple:
    """Restore the exact architecture and label order saved during training."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    required = {"model_state_dict", "model_config", "class_values"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f"Checkpoint must contain {sorted(required)}.")
    config = checkpoint["model_config"]
    class_values = checkpoint["class_values"]
    # Older checkpoints used native-size inputs; preserve that behavior.
    image_size = checkpoint.get("image_size")
    if image_size is not None and (
        not isinstance(image_size, (tuple, list)) or len(image_size) != 2
        or any(type(size) is not int or size < 1 for size in image_size)
    ):
        raise ValueError("Checkpoint image_size must contain two positive integers.")
    if (not isinstance(config, dict)
            or set(config) != {"in_channels", "num_classes", "base_channels"}
            or any(type(value) is not int or value < 1 for value in config.values())):
        raise ValueError("Checkpoint has an invalid model_config.")
    if (not isinstance(class_values, (tuple, list)) or not class_values
            or any(type(value) is not int for value in class_values)
            or len(set(class_values)) != len(class_values)
            or class_values[0] != 0
            or config["num_classes"] != len(class_values)
            or config["in_channels"] != 1):
        raise ValueError("Checkpoint must use one input channel and unique class IDs, "
                         "with background 0 first and one output channel per class.")
    model = UNet2D(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    # Release the second copy of the weights before measuring inference memory.
    del checkpoint
    return model, config, tuple(class_values), tuple(image_size) if image_size is not None else None


def save_comparison(image: torch.Tensor, target: torch.Tensor,
                    prediction: torch.Tensor, class_values: tuple,
                    filename: str, output_path: Path) -> None:
    """Show class maps plus foreground false positives/false negatives."""
    colors = ["black", "#56b4e9", "#e69f00", "#009e73", "#cc79a7", "#f0e442"]
    if len(class_values) > len(colors):
        colors = [plt.get_cmap("tab20")(i % 20) for i in range(len(class_values))]
        colors[0] = "black"
    cmap = ListedColormap(colors[:len(class_values)])
    norm = BoundaryNorm(torch.arange(len(class_values) + 1).numpy() - 0.5,
                        len(class_values))
    wrong = prediction != target
    false_positive = wrong & (prediction != 0)
    false_negative = wrong & (target != 0)
    overlay = torch.zeros((*target.shape, 4), dtype=torch.float32)
    overlay[false_positive] = torch.tensor([1.0, 0.0, 0.0, 0.7])
    overlay[false_negative] = torch.tensor([0.0, 0.3, 1.0, 0.7])
    overlay[false_positive & false_negative] = torch.tensor([1.0, 0.0, 1.0, 0.8])

    fig, axes = plt.subplots(1, 4, figsize=(13, 5), constrained_layout=True)
    axes[0].imshow(image[0], cmap="gray")
    axes[0].set_title("Input (normalized MRI)")
    axes[1].imshow(target, cmap=cmap, norm=norm)
    axes[1].set_title("Ground truth")
    label_plot = axes[2].imshow(prediction, cmap=cmap, norm=norm)
    axes[2].set_title("Prediction (argmax)")
    axes[3].imshow(image[0], cmap="gray")
    axes[3].imshow(overlay)
    axes[3].set_title("Foreground errors")
    for ax in axes:
        ax.axis("off")
    colorbar = fig.colorbar(label_plot, ax=axes[1:3], ticks=range(len(class_values)),
                            fraction=0.035, pad=0.02)
    colorbar.ax.set_yticklabels([str(value) for value in class_values])
    colorbar.set_label("Original label ID")
    axes[3].legend(handles=[Patch(color="red", label="False positive"),
                            Patch(color="#004cff", label="False negative"),
                            Patch(color="magenta", label="Wrong foreground class (both)")],
                   loc="lower center", bbox_to_anchor=(0.5, -0.23), fontsize=8)
    fig.suptitle(filename)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model, model_config, class_values, image_size = load_model(args.checkpoint, device)
    all_pairs = load_hipmri_splits(args.data_root)["test"]
    pairs = all_pairs
    if args.max_slices is not None and args.max_slices < len(pairs):
        indices = torch.linspace(0, len(pairs) - 1, args.max_slices).long().tolist()
        pairs = [pairs[index] for index in indices]
    dataset = Nifti2DDataset(pairs, class_values=class_values, image_size=image_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_count = min(args.num_figures, len(dataset))
    figure_indices = set(torch.linspace(0, len(dataset) - 1, figure_count).long().tolist())
    metrics = SegmentationMetrics(len(class_values))
    rows = []
    case_times = {}

    with torch.inference_mode():
        warmup_image = dataset[0][0].unsqueeze(0).to(device)
        for _ in range(3):
            model(warmup_image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        del warmup_image
        reset_peak_vram(device)
        for index, pair in enumerate(dataset.pairs):
            image, target = dataset[index]
            images = image.unsqueeze(0).to(device)
            logits, seconds = timed_inference(model, images)
            prediction = logits.argmax(dim=1).cpu()
            metrics.update(prediction, target.unsqueeze(0))
            # A case is one patient/week scan, represented here by its 2D slices.
            case_id = re.sub(r"_slice_\d+\.nii(?:\.gz)?$", "", pair.image_path.name)
            rows.append({"filename": pair.image_path.name, "patient_id": pair.patient_id,
                         "case_id": case_id, "forward_seconds": seconds})
            case_times.setdefault(case_id, []).append(seconds)
            if index in figure_indices:
                figure_name = pair.image_path.name.removesuffix(".gz").removesuffix(".nii")
                save_comparison(image, target, prediction[0], class_values,
                                pair.image_path.name,
                                args.output_dir / f"{figure_name}.png")
            del images, logits, prediction

    scores = metrics.compute()
    times = [row["forward_seconds"] for row in rows]
    case_rows = [{"case_id": case_id, "slices": len(values),
                  "total_forward_seconds": sum(values),
                  "mean_slice_forward_seconds": statistics.mean(values)}
                 for case_id, values in sorted(case_times.items())]
    result = {"checkpoint": str(args.checkpoint), "model_config": model_config,
              "class_values": list(class_values), "image_size": image_size, "scores": scores,
              "metric_aggregation": "Pixel counts pooled over evaluated slices; "
                                    "foreground mean excludes background and classes "
                                    "absent from both prediction and target.",
              "evaluated_slices": len(dataset), "total_test_slices": len(all_pairs),
              "subset_evaluation": len(dataset) < len(all_pairs),
              "parameter_count": count_parameters(model),
              "peak_vram_bytes": peak_vram_bytes(device),
              "peak_vram_scope": "Inference after warmup; peak PyTorch allocated tensor "
                                 "memory, excluding reserved but unused CUDA memory.",
              "device": str(device),
              "hardware": torch.cuda.get_device_name(device) if device.type == "cuda"
                          else platform.processor() or platform.machine(),
              "torch_version": str(torch.__version__),
              "timing": {"scope": "Batch size 1 model forward only after three warmup "
                                   "forwards; excludes disk I/O, transfer, argmax, metrics "
                                   "and figures. CUDA is synchronized. Case totals sum "
                                   "only evaluated slices.",
                         "mean_slice_seconds": statistics.mean(times),
                         "median_slice_seconds": statistics.median(times),
                         "total_forward_seconds": sum(times)},
              "cases": case_rows}
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "slice_timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "case_timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)

    print(f"Evaluated {len(dataset)}/{len(all_pairs)} test slices on {device}.")
    print("Label ID      Dice       IoU")
    for label, dice, iou in zip(class_values, scores["dice"], scores["iou"]):
        dice_text = "n/a" if dice is None else f"{dice:.4f}"
        iou_text = "n/a" if iou is None else f"{iou:.4f}"
        print(f"{label:8d} {dice_text:>9} {iou_text:>9}")
    mean_dice = scores["mean_foreground_dice"]
    print("Mean foreground Dice: " + ("n/a" if mean_dice is None else f"{mean_dice:.4f}"))
    print(f"Slice forward time: mean {statistics.mean(times):.6f} s, "
          f"median {statistics.median(times):.6f} s")
    for row in case_rows:
        print(f"{row['case_id']}: {row['total_forward_seconds']:.6f} s "
              f"across {row['slices']} evaluated slices")
    print(f"Saved metrics, timings and {figure_count} comparison figures to {args.output_dir}")


if __name__ == "__main__":
    main()
