"""Loss, split-level segmentation metrics, and basic PyTorch profiling.

Class index 0 is background. Metrics pool pixel counts across all supplied
slices, rather than averaging per-batch scores. These are slice-dataset scores,
not patient-level or reconstructed-volume scores.
"""

from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F


class DiceCELoss(nn.Module):
    """Cross-entropy over all classes plus soft Dice loss over foreground classes.

    Both terms have weight 1. Dice is averaged equally across foreground
    classes, pooling each class over the batch and spatial dimensions.
    Inputs are raw logits [B, C, H, W] and integer targets [B, H, W].
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        if smooth <= 0:
            raise ValueError("smooth must be positive.")
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if (logits.ndim != 4 or logits.shape[1] < 2
                or target.shape != (logits.shape[0], *logits.shape[2:])):
            raise ValueError("Expected logits [B, C>=2, H, W] and targets [B, H, W].")
        cross_entropy = F.cross_entropy(logits, target)
        # Float32 sums remain reliable if mixed precision is used later.
        probabilities = logits.float().softmax(dim=1)[:, 1:]
        one_hot = F.one_hot(target, num_classes=logits.shape[1])
        one_hot = one_hot.permute(0, 3, 1, 2).to(probabilities.dtype)[:, 1:]
        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return cross_entropy + (1 - dice.mean())


class SegmentationMetrics:
    """Accumulate a confusion matrix from predicted and true label maps.

    update() takes integer tensors [B, H, W]. A class absent in both prediction
    and truth is undefined (None), rather than a perfect score. False positives
    for a class absent from the truth give that class a score of zero.
    """

    def __init__(self, num_classes: int):
        if num_classes < 2:
            raise ValueError("num_classes must be at least two.")
        self.num_classes = num_classes
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.ndim != 3 or prediction.shape != target.shape:
            raise ValueError("Prediction and target must have matching [B, H, W] shapes.")
        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if prediction.dtype not in integer_types or target.dtype not in integer_types:
            raise ValueError("Prediction and target must contain integer class indices.")
        if prediction.numel() == 0:
            return
        prediction = prediction.reshape(-1).to(dtype=torch.int64)
        target = target.reshape(-1).to(device=prediction.device, dtype=torch.int64)
        if (prediction.min() < 0 or prediction.max() >= self.num_classes
                or target.min() < 0 or target.max() >= self.num_classes):
            raise ValueError("Class indices must be in [0, num_classes).")
        counts = torch.bincount(
            target * self.num_classes + prediction,
            minlength=self.num_classes ** 2,
        )
        self.confusion += counts.reshape(self.num_classes, self.num_classes).cpu()

    def compute(self) -> dict:
        """Return JSON-friendly per-class Dice/IoU and mean foreground Dice."""
        counts = self.confusion.to(dtype=torch.float64)
        intersection = counts.diag()
        actual = counts.sum(dim=1)
        predicted = counts.sum(dim=0)
        dice_denominator = actual + predicted
        union = actual + predicted - intersection
        dice = [
            float(2 * intersection[i] / dice_denominator[i])
            if dice_denominator[i] > 0 else None
            for i in range(self.num_classes)
        ]
        iou = [
            float(intersection[i] / union[i]) if union[i] > 0 else None
            for i in range(self.num_classes)
        ]
        foreground = [score for score in dice[1:] if score is not None]
        return {
            "dice": dice,
            "iou": iou,
            "mean_foreground_dice": sum(foreground) / len(foreground) if foreground else None,
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


def timed_inference(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Time one forward pass in seconds, synchronizing CUDA before and after.

    Caller must set model.eval() and move inputs to the model's device first.
    Disk loading, host-to-device transfer, argmax, and saving are excluded.
    Caller should perform untimed warm-up passes before benchmarking.
    """
    with torch.inference_mode():
        synchronize(images.device)
        start = perf_counter()
        logits = model(images)
        synchronize(images.device)
        elapsed = perf_counter() - start
    return logits, elapsed
