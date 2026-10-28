"""Loss, segmentation metrics, scan-level evaluation, profiling and chart style.

Class index 0 is background. Scores are computed per scan (one weekly MR volume) from
a confusion matrix:
- Dice = 2TP / (2TP + FP + FN) and IoU = TP / (TP + FP + FN), for every class.
- A class absent from both prediction and truth is undefined (None), not a perfect
  score, and is left out of every mean. Predicting a class the scan lacks scores 0.
- Summaries average each patient's scans first, then report the mean and sample SD
  across patients, so a patient with eight weekly scans counts once. Scores pooled
  over all voxels are kept for reference.
At test time, predictions are resampled to the full-resolution labels before scoring
(evaluate_scans), so the 2D and 3D models are scored on exactly the same voxels.
"""

from dataclasses import dataclass
from math import prod
from statistics import mean, median, stdev
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from dataset import LABEL_NAMES, Scan, load_scan
from modules import ImprovedUNet3D, UNet2D


@dataclass(frozen=True)
class ModelSpec:
    """How to build, run and label one model type."""

    cls: type
    slicewise: bool  # True: a 2D model applied to each axial slice of a scan
    label: str
    color: str


MODELS = {
    "unet2d": ModelSpec(UNet2D, True, "2D U-Net", "#2a78d6"),
    "improved3d": ModelSpec(ImprovedUNet3D, False, "3D Improved U-Net", "#eb6834"),
}

# Chart style shared by train.py and predict.py. The model and organ colours are
# colour-blind-checked sets; text always uses the ink colours, never a data colour.
SURFACE, INK, INK_SECONDARY, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
LABEL_COLORS = {"body": INK_SECONDARY, "bone": INK_MUTED, "bladder": "#6250d6",
                "rectum": "#1baf7a", "prostate": "#eda100"}
CHART_RC = {
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY, "text.color": INK, "legend.frameon": False,
    "font.size": 9, "lines.linewidth": 2, "lines.solid_capstyle": "round",
}


class DiceCELoss(nn.Module):
    """Cross-entropy over all classes plus soft Dice loss over foreground classes.

    Logits are [B, C, ...] and integer targets [B, ...], so one loss serves 2D slices
    and 3D volumes. Each foreground class is pooled over the batch and spatial axes.
    Only classes present in the targets enter the Dice mean: an absent class would add
    an almost constant 1/(C-1), making the loss depend on batch composition rather
    than on the predictions. Cross-entropy still penalises predicting absent classes.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        if smooth <= 0:
            raise ValueError("smooth must be positive.")
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if (logits.ndim < 3 or logits.shape[1] < 2
                or target.shape != (logits.shape[0], *logits.shape[2:])):
            raise ValueError("Expected logits [B, C>=2, ...] and targets [B, ...].")
        cross_entropy = F.cross_entropy(logits, target)
        # Float32 sums remain reliable if mixed precision is used later.
        probabilities = logits.float().softmax(dim=1)
        one_hot = torch.zeros_like(probabilities).scatter_(1, target.unsqueeze(1), 1.0)
        dims = (0, *range(2, logits.ndim))
        intersection = (probabilities * one_hot).sum(dim=dims)[1:]
        truth = one_hot.sum(dim=dims)[1:]
        dice = (2 * intersection + self.smooth) / (probabilities.sum(dim=dims)[1:] + truth
                                                   + self.smooth)
        present = truth > 0
        dice_loss = (1 - dice[present]).mean() if present.any() else dice.new_zeros(())
        return cross_entropy + dice_loss


@torch.no_grad()
def confusion_matrix(prediction: torch.Tensor, target: torch.Tensor,
                     num_classes: int) -> torch.Tensor:
    """Count voxels per (true class, predicted class) for label maps of any shape.

    Rows are the ground truth and columns the prediction. Returns int64 on the CPU.
    """
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have the same shape.")
    integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if prediction.dtype not in integer_types or target.dtype not in integer_types:
        raise ValueError("Prediction and target must contain integer class indices.")
    prediction = prediction.reshape(-1).to(dtype=torch.int64)
    target = target.reshape(-1).to(device=prediction.device, dtype=torch.int64)
    if prediction.numel() and (prediction.min() < 0 or prediction.max() >= num_classes
                               or target.min() < 0 or target.max() >= num_classes):
        raise ValueError("Class indices must be in [0, num_classes).")
    counts = torch.bincount(target * num_classes + prediction, minlength=num_classes ** 2)
    return counts.reshape(num_classes, num_classes).cpu()


def _defined_mean(values: list) -> float | None:
    defined = [value for value in values if value is not None]
    return mean(defined) if defined else None


def scores_from_confusion(confusion: torch.Tensor) -> dict:
    """Per-class Dice and IoU, plus their means over the foreground classes 1..C-1.

    Dice = 2TP / (|truth| + |prediction|) and IoU = TP / (|truth| + |prediction| - TP).
    """
    counts = confusion.to(dtype=torch.float64)
    tp = counts.diag()
    total = counts.sum(dim=1) + counts.sum(dim=0)
    dice = [float(2 * tp[i] / total[i]) if total[i] > 0 else None for i in range(len(tp))]
    iou = [float(tp[i] / (total[i] - tp[i])) if total[i] > 0 else None for i in range(len(tp))]
    return {"dice": dice, "iou": iou, "mean_foreground_dice": _defined_mean(dice[1:]),
            "mean_foreground_iou": _defined_mean(iou[1:])}


class SegmentationMetrics:
    """Accumulate a confusion matrix over batches; compute() pools all voxels seen."""

    def __init__(self, num_classes: int):
        if num_classes < 2:
            raise ValueError("num_classes must be at least two.")
        self.num_classes = num_classes
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        self.confusion += confusion_matrix(prediction, target, self.num_classes)

    def compute(self) -> dict:
        return scores_from_confusion(self.confusion)


@torch.no_grad()
def region_scores(prediction: torch.Tensor, target: torch.Tensor, label: int) -> dict:
    """Score one label along the axial axis D of [D, H, W] label maps.

    The label's ground-truth extent (first to last axial slice containing it) is split
    into inferior, middle and superior thirds; for the prostate these are the apex,
    mid-gland and base. Each third gets a Dice over its slices, and predictions of
    the label on slices outside the extent are counted as false-positive voxels below
    and above it. The per-slice Dice inside the extent is returned with each slice's
    relative position (0 = most inferior, 1 = most superior).
    """
    truth = (target == label).flatten(1)
    predicted = (prediction == label).flatten(1)
    present = truth.any(dim=1).nonzero().flatten()
    if present.numel() == 0:
        return {"thirds": [None, None, None], "profile": [],
                "fp_below_voxels": None, "fp_above_voxels": None}
    first, last = int(present[0]), int(present[-1])
    tp = (truth & predicted).sum(dim=1)
    total = truth.sum(dim=1) + predicted.sum(dim=1)

    def dice(slices) -> float | None:
        denominator = total[slices].sum()
        return float(2 * tp[slices].sum() / denominator) if denominator > 0 else None

    span = max(last - first, 1)
    return {
        "thirds": [dice(part) for part in torch.arange(first, last + 1).tensor_split(3)],
        "profile": [((depth - first) / span, dice(slice(depth, depth + 1)))
                    for depth in range(first, last + 1)],
        "fp_below_voxels": int(predicted[:first].sum()),
        "fp_above_voxels": int(predicted[last + 1:].sum()),
    }


def patient_summary(values: list, patients: list[str]) -> dict:
    """Average each patient's defined values, then summarise across patients.

    Returns the mean and sample SD across patients (SD needs two patients) and the
    per-patient means. Undefined values (None) are skipped.
    """
    grouped = {}
    for value, patient in zip(values, patients):
        if value is not None:
            grouped.setdefault(patient, []).append(value)
    per_patient = {patient: mean(items) for patient, items in sorted(grouped.items())}
    scores = list(per_patient.values())
    return {"mean": mean(scores) if scores else None,
            "sd": stdev(scores) if len(scores) > 1 else None, "per_patient": per_patient}


def format_summary(summary: dict, digits: int = 3) -> str:
    """'mean ± SD' text for a patient_summary() result."""
    if summary["mean"] is None:
        return "n/a"
    text = f"{summary['mean']:.{digits}f}"
    return text if summary["sd"] is None else f"{text} ± {summary['sd']:.{digits}f}"


@torch.no_grad()
def predict_volume(model: nn.Module, image: torch.Tensor, *, slicewise: bool,
                   slice_batch: int = 32) -> torch.Tensor:
    """Logits [1, C, D, H, W] for one scan image [1, D, H, W] on the model's device.

    A 3D model sees the whole volume. A slice-wise 2D model predicts every axial slice
    independently and its slice logits are stacked back along D, so both models are
    scored on the same volume. The caller sets model.eval().
    """
    if not slicewise:
        return model(image.unsqueeze(0))
    slices = image[0].unsqueeze(1)
    logits = torch.cat([model(slices[start:start + slice_batch])
                        for start in range(0, len(slices), slice_batch)])
    return logits.permute(1, 0, 2, 3).unsqueeze(0)


@torch.no_grad()
def predict_scan(model: nn.Module, scan: Scan, *, slicewise: bool, downsample: int,
                 device: torch.device) -> dict:
    """Predict one scan at the model's working resolution and return it at full resolution.

    Class probabilities are resampled (trilinear) to the native label grid before the
    argmax. Returns the native MR "image" and "target" [D, H, W], the "prediction"
    (uint8, CPU), the voxel "spacing" in mm, the CUDA-synchronized forward "seconds"
    (model only) and the forward's "peak_vram_bytes" (None on CPU).
    """
    image = load_scan(scan, downsample)[0].to(device)
    native_image, target, spacing = load_scan(scan, 1)
    reset_peak_vram(device)
    synchronize(device)
    start = perf_counter()
    logits = predict_volume(model, image, slicewise=slicewise)
    synchronize(device)
    seconds = perf_counter() - start
    peak = peak_vram_bytes(device)
    probabilities = logits.softmax(dim=1)
    if probabilities.shape[2:] != target.shape:
        probabilities = F.interpolate(probabilities, size=tuple(target.shape),
                                      mode="trilinear", align_corners=False)
    prediction = probabilities.argmax(dim=1)[0].to(torch.uint8).cpu()
    return {"image": native_image[0], "target": target, "prediction": prediction,
            "spacing": spacing, "seconds": seconds, "peak_vram_bytes": peak}


def evaluate_scans(model: nn.Module, scans: list[Scan], *, slicewise: bool, downsample: int,
                   device: torch.device, organ: int) -> list[dict]:
    """Score one model on whole scans against their full-resolution labels.

    Returns one JSON-ready record per scan: patient, week, confusion matrix, per-class
    Dice/IoU with foreground means, region_scores() for the organ label, voxel volume
    in mL, forward seconds and forward peak CUDA memory.
    """
    model.eval()
    # Warm-up, so lazy CUDA set-up and kernel selection are not timed.
    predict_volume(model, load_scan(scans[0], downsample)[0].to(device), slicewise=slicewise)
    records = []
    for scan in scans:
        result = predict_scan(model, scan, slicewise=slicewise, downsample=downsample,
                              device=device)
        confusion = confusion_matrix(result["prediction"], result["target"], len(LABEL_NAMES))
        records.append({
            "patient_id": scan.patient_id, "week": scan.week,
            "voxel_ml": prod(result["spacing"]) / 1000,
            "forward_seconds": result["seconds"], "peak_vram_bytes": result["peak_vram_bytes"],
            "confusion": confusion.tolist(), **scores_from_confusion(confusion),
            "organ": {"label": organ,
                      **region_scores(result["prediction"], result["target"], organ)},
        })
    return records


def summarize(records: list[dict], profile_bins: int = 10) -> dict:
    """Patient-level summaries (mean, SD, per-patient means) and pooled scores of records.

    Organ false positives are in mL. organ_predicted_in_class_ml[k] is the organ volume
    predicted inside true class k (k == organ gives the true positives). organ_profile
    bins the per-slice organ Dice by relative position along its axial extent.
    """
    patients = [record["patient_id"] for record in records]
    organ = records[0]["organ"]["label"]
    num_classes = len(records[0]["dice"])

    def by_patient(values: list) -> dict:
        return patient_summary(values, patients)

    def millilitres(record: dict, voxels: int | None) -> float | None:
        return None if voxels is None else voxels * record["voxel_ml"]

    profile = []
    for index in range(profile_bins):
        values = []
        for record in records:
            inside = [dice for position, dice in record["organ"]["profile"]
                      if dice is not None and min(int(position * profile_bins), profile_bins - 1) == index]
            values.append(mean(inside) if inside else None)
        profile.append({"center": (index + 0.5) / profile_bins, **by_patient(values)})
    peaks = [record["peak_vram_bytes"] for record in records if record["peak_vram_bytes"] is not None]
    seconds = [record["forward_seconds"] for record in records]
    return {
        "patients": len(set(patients)), "scans": len(records),
        "dice": [by_patient([r["dice"][c] for r in records]) for c in range(num_classes)],
        "iou": [by_patient([r["iou"][c] for r in records]) for c in range(num_classes)],
        "mean_foreground_dice": by_patient([r["mean_foreground_dice"] for r in records]),
        "mean_foreground_iou": by_patient([r["mean_foreground_iou"] for r in records]),
        "pooled": scores_from_confusion(torch.tensor([r["confusion"] for r in records]).sum(0)),
        "organ_label": organ,
        "organ_thirds_dice": [by_patient([r["organ"]["thirds"][k] for r in records])
                              for k in range(3)],
        "organ_fp_below_ml": by_patient([millilitres(r, r["organ"]["fp_below_voxels"])
                                         for r in records]),
        "organ_fp_above_ml": by_patient([millilitres(r, r["organ"]["fp_above_voxels"])
                                         for r in records]),
        "organ_predicted_in_class_ml": [by_patient([r["confusion"][k][organ] * r["voxel_ml"]
                                                    for r in records]) for k in range(num_classes)],
        "organ_profile": profile,
        "forward_seconds": {"mean": mean(seconds), "median": median(seconds)},
        "peak_vram_bytes": max(peaks) if peaks else None,
    }


def count_parameters(model: nn.Module) -> int:
    """Count all model parameters, including any frozen parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def synchronize(device: torch.device | str) -> None:
    """Wait for CUDA work on this device; CPU requires no synchronization."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_vram(device: torch.device | str) -> None:
    """Start a new CUDA peak-allocation measurement window (CPU: no-op)."""
    device = torch.device(device)
    if device.type == "cuda":
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def peak_vram_bytes(device: torch.device | str) -> int | None:
    """Peak PyTorch CUDA tensor allocation in bytes; unavailable on CPU.

    This is max_memory_allocated, not total device memory or reserved memory.
    """
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    return None
