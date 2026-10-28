"""Evaluate trained models on the test patients and compare them in tables and figures.

    python predict.py --checkpoints runs/unet2d/best.pt runs/improved3d/best.pt

Each checkpoint predicts every test scan at its own working resolution, and its
predictions are resampled to the full-resolution labels before scoring, so every model
is scored on the same voxels (utils.evaluate_scans). Tables give the mean ± SD over test
patients (each patient's scans averaged first), with scores pooled over all test voxels
in brackets. Saved to --output-dir: metrics.json, scan_scores.csv, class_scores.png,
prostate_regions.png and case_<patient>_week<n>.png comparison panels.
"""

import argparse
import csv
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import torch

from dataset import (DEFAULT_ROOT, IMAGE_DIR, LABEL_DIR, LABEL_NAMES, SPLIT_FILE, Scan,
                     discover_scans, load_split, split_scans)
from utils import (CHART_RC, LABEL_COLORS, MODELS, SURFACE, count_parameters, evaluate_scans,
                   format_summary, predict_scan, summarize)


PROSTATE = LABEL_NAMES.index("prostate")
BLADDER = LABEL_NAMES.index("bladder")
RECTUM = LABEL_NAMES.index("rectum")
ORGANS = ("bladder", "rectum", "prostate")
# Categorical slots for up to three models; each model type keeps its own colour first.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a")
OUTLINE = [patheffects.withStroke(linewidth=2.4, foreground="black")]


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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                        help="One to three best.pt files written by train.py.")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--label-dir", default=LABEL_DIR)
    parser.add_argument("--split-file", type=Path, default=SPLIT_FILE)
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--num-cases", type=_nonnegative_int, default=3,
                        help="Comparison figures for scans spread over the prostate-Dice "
                             "difference between the first and last model.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-scans", type=_positive_int,
                        help="Evaluate an evenly spaced subset of test scans for a smoke check.")
    args = parser.parse_args()
    if len(args.checkpoints) > len(PALETTE):
        parser.error(f"Compare at most {len(PALETTE)} checkpoints.")
    return args


def load_model(path: Path, device: torch.device, test_patients: list[str]) -> dict:
    """Restore a train.py checkpoint; refuse one that saw a test patient in training."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {"model", "model_state_dict", "model_config", "downsample", "split"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f"{path} is not a train.py checkpoint (needs {sorted(required)}).")
    if checkpoint["model"] not in MODELS:
        raise ValueError(f"{path} has unknown model type {checkpoint['model']!r}.")
    seen = set(checkpoint["split"]["train"]) | set(checkpoint["split"]["val"])
    leaked = sorted(seen & set(test_patients))
    if leaked:
        raise ValueError(f"{path} was trained or validated on test patients {leaked[:5]}.")
    spec = MODELS[checkpoint["model"]]
    model = spec.cls(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {"path": path, "name": checkpoint["model"], "spec": spec, "model": model,
            "downsample": checkpoint["downsample"], "epoch": checkpoint.get("epoch")}


def _assign_labels_and_colors(entries: list[dict]) -> None:
    """Label models by type (plus run folder if repeated) and give each its own colour."""
    used = []
    for entry in entries:
        repeated = sum(other["name"] == entry["name"] for other in entries) > 1
        entry["label"] = (f"{entry['spec'].label} ({entry['path'].parent.name})" if repeated
                          else entry["spec"].label)
        color = entry["spec"].color
        if color in used:
            color = next(slot for slot in PALETTE if slot not in used)
        used.append(color)
        entry["color"] = color


def _pooled(result: dict, metric: str, channel: int) -> str:
    value = result["summary"]["pooled"][metric][channel]
    return "n/a" if value is None else f"{value:.3f}"


def print_report(results: list[dict], total_scans: int) -> None:
    """Print per-class Dice and IoU, prostate scores along the axial axis and costs."""
    first = results[0]["summary"]
    print(f"\nTest set: {first['patients']} patients, {first['scans']} of {total_scans} scans, "
          "scored against the full-resolution labels.")
    print("Mean ± SD over patients (each patient's scans averaged first); "
          "[pooled over all test voxels].")
    width = max(28, *(len(result["label"]) + 2 for result in results))

    def row(name: str, cells: list[str]) -> None:
        print(f"  {name:<26}" + "".join(f"{cell:>{width}}" for cell in cells))

    row("", [result["label"] for result in results])
    for metric, title in (("dice", "Dice"), ("iou", "IoU")):
        print(title)
        for channel, name in enumerate(LABEL_NAMES[1:], start=1):
            row(name, [f"{format_summary(r['summary'][metric][channel])} "
                       f"[{_pooled(r, metric, channel)}]" for r in results])
        row("foreground mean", [
            f"{format_summary(r['summary'][f'mean_foreground_{metric}'])} "
            f"[{r['summary']['pooled'][f'mean_foreground_{metric}']:.3f}]" for r in results])
    print("Prostate Dice along the head-foot axis")
    for index, name in enumerate(("apex (inferior third)", "mid-gland", "base (superior third)")):
        row(name, [format_summary(r["summary"]["organ_thirds_dice"][index]) for r in results])
    print("Prostate false positives (mL)")
    for name, pick in (("below the apex", lambda s: s["organ_fp_below_ml"]),
                       ("above the base", lambda s: s["organ_fp_above_ml"]),
                       ("inside the bladder", lambda s: s["organ_predicted_in_class_ml"][BLADDER]),
                       ("inside the rectum", lambda s: s["organ_predicted_in_class_ml"][RECTUM])):
        row(name, [format_summary(pick(r["summary"]), digits=2) for r in results])
    print("Cost")
    row("parameters", [f"{r['parameters']:,}" for r in results])
    row("peak forward VRAM (MiB)", [
        "n/a (CPU)" if r["summary"]["peak_vram_bytes"] is None
        else f"{r['summary']['peak_vram_bytes'] / 2 ** 20:.0f}" for r in results])
    row("forward time per scan (s)", [
        f"{r['summary']['forward_seconds']['median']:.3f} median" for r in results])
    row("working grid", [f"downsampled x{r['downsample']}" for r in results])


def write_scan_csv(results: list[dict], path: Path) -> None:
    """One row per model and test scan; empty cells are undefined scores."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "patient_id", "week",
                         *(f"dice_{name}" for name in LABEL_NAMES),
                         *(f"iou_{name}" for name in LABEL_NAMES),
                         "prostate_dice_apex_third", "prostate_dice_mid_third",
                         "prostate_dice_base_third", "prostate_fp_below_apex_ml",
                         "prostate_fp_above_base_ml", "prostate_in_bladder_ml",
                         "prostate_in_rectum_ml", "forward_seconds"])
        for result in results:
            for record in result["scans"]:
                organ, voxel_ml = record["organ"], record["voxel_ml"]
                below, above = organ["fp_below_voxels"], organ["fp_above_voxels"]
                writer.writerow([
                    result["label"], record["patient_id"], record["week"],
                    *record["dice"], *record["iou"], *organ["thirds"],
                    None if below is None else below * voxel_ml,
                    None if above is None else above * voxel_ml,
                    record["confusion"][BLADDER][PROSTATE] * voxel_ml,
                    record["confusion"][RECTUM][PROSTATE] * voxel_ml,
                    record["forward_seconds"],
                ])


def _bars_with_patients(ax: plt.Axes, results: list[dict], groups: list[str], pick) -> None:
    """Grouped bars of the mean over patients, with one ringed dot per patient.

    pick(summary, group_index) returns a patient_summary() result.
    """
    width, gap = 0.16, 0.02
    for number, result in enumerate(results):
        offset = (number - (len(results) - 1) / 2) * (width + gap)
        for index in range(len(groups)):
            summary = pick(result["summary"], index)
            if summary["mean"] is None:
                continue
            x = index + offset
            ax.bar(x, summary["mean"], width, color=result["color"],
                   label=result["label"] if index == 0 else None)
            points = list(summary["per_patient"].values())
            spread = (torch.linspace(-width / 4, width / 4, len(points)).tolist()
                      if len(points) > 1 else [0.0])
            ax.scatter([x + step for step in spread], points, s=22, color=result["color"],
                       edgecolors=SURFACE, linewidths=1.2, zorder=3)
    ax.set_xticks(range(len(groups)), groups)


def plot_class_scores(results: list[dict], path: Path) -> None:
    """Dice and IoU per foreground class: bars are patient means, dots are patients."""
    classes = list(range(1, len(LABEL_NAMES)))
    with plt.rc_context(CHART_RC):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True, layout="constrained")
        for ax, metric, title in zip(axes, ("dice", "iou"), ("Dice", "IoU")):
            _bars_with_patients(ax, results, [LABEL_NAMES[c] for c in classes],
                                lambda summary, index: summary[metric][classes[index]])
            ax.set(title=f"{title} per class on the test patients", ylim=(0, 1))
        axes[0].set_ylabel("Score (bar: mean over patients; dot: one patient)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=len(results))
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_prostate(results: list[dict], path: Path) -> None:
    """Prostate Dice along the head-foot axis, by third, and false-positive volumes."""
    false_positives = (("below apex", lambda s: s["organ_fp_below_ml"]),
                       ("above base", lambda s: s["organ_fp_above_ml"]),
                       ("in bladder", lambda s: s["organ_predicted_in_class_ml"][BLADDER]),
                       ("in rectum", lambda s: s["organ_predicted_in_class_ml"][RECTUM]))
    with plt.rc_context(CHART_RC):
        fig, (profile_ax, thirds_ax, fp_ax) = plt.subplots(
            1, 3, figsize=(15, 4.1), layout="constrained",
            gridspec_kw={"width_ratios": [1.35, 1, 1.25]})
        for result in results:
            bins = result["summary"]["organ_profile"]
            centers = [item["center"] for item in bins]
            means = [float("nan") if item["mean"] is None else item["mean"] for item in bins]
            sds = [item["sd"] or 0.0 for item in bins]
            profile_ax.fill_between(centers, [m - s for m, s in zip(means, sds)],
                                    [m + s for m, s in zip(means, sds)],
                                    color=result["color"], alpha=0.1, linewidth=0)
            profile_ax.plot(centers, means, color=result["color"], marker="o", markersize=4,
                            markeredgecolor=SURFACE, markeredgewidth=1.2, label=result["label"])
        profile_ax.set(xlim=(0, 1), ylim=(0, 1), title="Prostate Dice per axial slice",
                       xlabel="Position within the true prostate extent",
                       ylabel="Dice (line: mean over patients; band: ± SD)")
        profile_ax.set_xticks([0, 0.5, 1], ["apex (inferior)", "mid-gland", "base (superior)"])
        _bars_with_patients(thirds_ax, results, ["apex third", "mid-gland", "base third"],
                            lambda summary, index: summary["organ_thirds_dice"][index])
        thirds_ax.set(ylim=(0, 1), title="Prostate Dice by third", ylabel="Dice")
        _bars_with_patients(fp_ax, results, [name for name, _ in false_positives],
                            lambda summary, index: false_positives[index][1](summary))
        fp_ax.set(title="Predicted prostate outside the true prostate", ylabel="Volume (mL)")
        handles, labels = profile_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=len(results))
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_training(entries: list[dict], path: Path) -> bool:
    """Validation Dice against cumulative training time, read from each run's history.json.

    The selected checkpoint is marked. Returns False when a history file is missing.
    """
    histories = []
    for entry in entries:
        history_path = entry["path"].parent / "history.json"
        if not history_path.exists():
            return False
        histories.append(json.loads(history_path.read_text()))
    total_hours = max(sum(row["seconds"] for row in history) for history in histories) / 3600
    scale, unit = (3600, "hours") if total_hours >= 2 else (60, "minutes")
    with plt.rc_context(CHART_RC):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, layout="constrained")
        for entry, history in zip(entries, histories):
            elapsed, times = 0.0, []
            for row in history:
                elapsed += row["seconds"] / scale
                times.append(elapsed)
            for ax, values in zip(axes, ([row["val"]["mean_foreground_dice"] for row in history],
                                         [row["val"]["dice"][PROSTATE] for row in history])):
                ax.plot(times, values, color=entry["color"], label=entry["label"])
                if entry["epoch"] is not None and entry["epoch"] <= len(history):
                    best = entry["epoch"] - 1
                    ax.plot(times[best], values[best], linestyle="none", marker="o", markersize=5,
                            color=entry["color"], markeredgecolor=SURFACE, markeredgewidth=1.2)
        axes[0].set(title="Validation mean foreground Dice", ylim=(0, 1),
                    ylabel="Dice (mean over validation patients)")
        axes[1].set(title="Validation prostate Dice")
        for ax in axes:
            ax.set_xlabel(f"Training time including validation ({unit}); dot: chosen checkpoint")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=len(entries))
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return True


def choose_cases(results: list[dict], count: int) -> list[tuple[int, str]]:
    """Pick scans spread over the prostate-Dice difference (last minus first model),
    or over the prostate Dice itself for one model, always including both extremes."""
    first = [record["dice"][PROSTATE] or 0.0 for record in results[0]["scans"]]
    if len(results) > 1:
        last = [record["dice"][PROSTATE] or 0.0 for record in results[-1]["scans"]]
        key = [b - a for a, b in zip(first, last)]
        low, high = f"{results[0]['label']} furthest ahead", f"{results[-1]['label']} furthest ahead"
        measure = "prostate-Dice difference"
    else:
        key, low, high = first, "lowest prostate Dice", "highest prostate Dice"
        measure = "prostate Dice"
    if count == 0 or not key:
        return []
    order = sorted(range(len(key)), key=key.__getitem__)
    ranks = sorted({round(i * (len(order) - 1) / max(count - 1, 1)) for i in range(count)})
    return [(order[rank], low if rank == 0 else high if rank == len(order) - 1
             else f"rank {rank + 1} of {len(order)} by {measure}") for rank in ranks]


def _window(center: int, half: int, size: int) -> tuple[int, int]:
    """Indices [start, stop) of a window of up to 2 * half voxels around center."""
    start = min(max(center - half, 0), max(size - 2 * half, 0))
    return start, min(start + 2 * half, size)


def _dice(prediction: torch.Tensor, target: torch.Tensor) -> str:
    predicted, truth = prediction == PROSTATE, target == PROSTATE
    total = int(predicted.sum() + truth.sum())
    return f"{2 * int((predicted & truth).sum()) / total:.2f}" if total else "n/a"


def _overlay(labels: torch.Tensor) -> torch.Tensor:
    """RGBA image with the organs filled in their colours."""
    rgba = torch.zeros((*labels.shape, 4))
    for organ in ORGANS:
        rgba[labels == LABEL_NAMES.index(organ)] = torch.tensor([*to_rgb(LABEL_COLORS[organ]), 0.5])
    return rgba


def _draw(ax: plt.Axes, mri: torch.Tensor, labels: torch.Tensor, outline: torch.Tensor | None,
          window: tuple[float, float], aspect: float) -> None:
    ax.imshow(mri, cmap="gray", vmin=window[0], vmax=window[1], aspect=aspect,
              interpolation="nearest")
    ax.imshow(_overlay(labels), aspect=aspect, interpolation="nearest")
    if outline is not None:
        for organ in ORGANS:
            mask = outline == LABEL_NAMES.index(organ)
            if mask.any():
                ax.contour(mask.float(), levels=[0.5], colors="white",
                           linewidths=1.0).set(path_effects=OUTLINE)
    ax.set(xticks=[], yticks=[])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _annotate(ax: plt.Axes, text: str) -> None:
    ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
            color="white", bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 2})


def plot_case(scan: Scan, outputs: list[dict], results: list[dict], subtitle: str,
              path: Path) -> None:
    """Apex, mid-gland and base axial slices plus a sagittal cut through the prostate.

    Columns are the ground truth and each model's prediction with organs filled; model
    columns also outline the true organs in white, so every gap between the fill and the
    outline is an error.
    """
    image, target, spacing = outputs[0]["image"], outputs[0]["target"], outputs[0]["spacing"]
    prostate = target == PROSTATE
    present = prostate.flatten(1).any(dim=1).nonzero().flatten()
    if present.numel() == 0:
        print(f"Skipped {path.name}: the ground truth has no prostate.")
        return
    first, last = int(present[0]), int(present[-1])
    thirds = torch.arange(first, last + 1).tensor_split(3)
    rows = [(name, int(part[len(part) // 2]))
            for name, part in zip(("Apex", "Mid-gland", "Base"), thirds) if len(part)]
    _, prostate_h, prostate_w = prostate.nonzero(as_tuple=True)
    centre_w = int(prostate_w.float().mean())
    h0, h1 = _window(int(prostate_h.float().mean()), 40, target.shape[1])
    w0, w1 = _window(centre_w, 40, target.shape[2])
    d0, d1 = max(first - 12, 0), min(last + 13, target.shape[0])
    sample = image.flatten()[::7]
    window = (float(sample.quantile(0.01)), float(sample.quantile(0.995)))
    maps = [target, *(output["prediction"] for output in outputs)]
    columns = ["Ground truth", *(result["label"] for result in results)]
    # Physical height of the sagittal panel relative to the (square) axial panels.
    sagittal = (d1 - d0) * spacing[0] / ((h1 - h0) * spacing[1])

    with plt.rc_context(CHART_RC):
        fig, axes = plt.subplots(len(rows) + 1, len(columns), squeeze=False, layout="constrained",
                                 figsize=(2.6 * len(columns), 2.6 * (len(rows) + sagittal) + 1.0),
                                 gridspec_kw={"height_ratios": [1] * len(rows) + [sagittal]})
        for row, (name, depth) in enumerate(rows):
            # Radiological view: anterior at the top, the patient's right on the left.
            views = [labels[depth, h0:h1, w0:w1].flip(0).flip(1) for labels in maps]
            mri = image[depth, h0:h1, w0:w1].flip(0).flip(1)
            for column, view in enumerate(views):
                _draw(axes[row, column], mri, view, views[0] if column else None, window,
                      spacing[1] / spacing[2])
                if column:
                    _annotate(axes[row, column], f"slice Dice {_dice(maps[column][depth], target[depth])}")
            axes[row, 0].set_ylabel(f"{name}\naxial slice {depth}")
        # Sagittal cut through the prostate centre: superior at the top, anterior on the left.
        views = [labels[d0:d1, h0:h1, centre_w].flip(0).flip(1) for labels in maps]
        mri = image[d0:d1, h0:h1, centre_w].flip(0).flip(1)
        for column, view in enumerate(views):
            _draw(axes[-1, column], mri, view, views[0] if column else None, window,
                  spacing[0] / spacing[1])
            if column:
                _annotate(axes[-1, column], f"scan Dice {_dice(maps[column], target)}")
        axes[-1, 0].set_ylabel("Sagittal, through\nthe prostate centre")
        for column, name in enumerate(columns):
            axes[0, column].set_title(name)
        handles = [Patch(color=LABEL_COLORS[organ], alpha=0.5, label=organ) for organ in ORGANS]
        handles.append(Line2D([], [], color="white", linewidth=1.5, path_effects=OUTLINE,
                              label="true organ outline"))
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
        fig.suptitle(f"Patient {scan.patient_id}, week {scan.week}\n{subtitle}", fontsize=10)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    split = load_split(args.split_file)
    scans = split_scans(discover_scans(args.data_root, args.image_dir, args.label_dir), split)["test"]
    total_scans = len(scans)
    if args.max_scans is not None and args.max_scans < total_scans:
        indices = torch.linspace(0, total_scans - 1, args.max_scans).long().tolist()
        scans = [scans[index] for index in indices]
        print("SMOKE CHECK: --max-scans evaluates a subset; do not report these results.")
    entries = [load_model(path, device, split["test"]) for path in args.checkpoints]
    _assign_labels_and_colors(entries)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for entry in entries:
        print(f"Evaluating {entry['label']} ({entry['path']}) on {len(scans)} test scans...")
        records = evaluate_scans(entry["model"], scans, slicewise=entry["spec"].slicewise,
                                 downsample=entry["downsample"], device=device, organ=PROSTATE)
        results.append({"label": entry["label"], "color": entry["color"], "model": entry["name"],
                        "checkpoint": str(entry["path"]), "epoch": entry["epoch"],
                        "downsample": entry["downsample"],
                        "parameters": count_parameters(entry["model"]),
                        "summary": summarize(records), "scans": records})
    print_report(results, total_scans)

    hardware = (torch.cuda.get_device_name(device) if device.type == "cuda"
                else platform.processor() or platform.machine())
    metrics = {
        "split_file": str(args.split_file), "test_patients": split["test"],
        "evaluated_scans": len(scans), "total_test_scans": total_scans,
        "subset_evaluation": len(scans) < total_scans, "device": str(device),
        "hardware": hardware, "torch_version": str(torch.__version__),
        "scope": "Per-scan scores against the full-resolution labels. Summaries average each "
                 "patient's scans, then give the mean and sample SD over patients; 'pooled' "
                 "counts all test voxels together. Forward time covers the model only "
                 "(CUDA-synchronized, after one warm-up scan); peak VRAM is the allocated "
                 "CUDA memory during a forward pass, including the weights.",
        "models": results,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    write_scan_csv(results, args.output_dir / "scan_scores.csv")
    plot_class_scores(results, args.output_dir / "class_scores.png")
    plot_prostate(results, args.output_dir / "prostate_regions.png")
    figures = ["class_scores.png", "prostate_regions.png"]
    if plot_training(entries, args.output_dir / "training_comparison.png"):
        figures.append("training_comparison.png")
    for index, reason in choose_cases(results, args.num_cases):
        scan = scans[index]
        outputs = [predict_scan(entry["model"], scan, slicewise=entry["spec"].slicewise,
                                downsample=entry["downsample"], device=device)
                   for entry in entries]
        dice_text = ", ".join(f"{result['label']} {_dice(output['prediction'], output['target'])}"
                              for result, output in zip(results, outputs))
        name = f"case_{scan.patient_id}_week{scan.week}.png"
        plot_case(scan, outputs, results, f"{reason}. Prostate Dice per scan: {dice_text}",
                  args.output_dir / name)
        figures.append(name)
    print(f"\nSaved metrics.json, scan_scores.csv and {', '.join(figures)} to {args.output_dir}")


if __name__ == "__main__":
    main()
