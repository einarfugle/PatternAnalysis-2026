"""Load the pre-extracted HipMRI 2D slices for the baseline U-Net.

Use load_hipmri_splits() to pair the existing train/validation/test folders,
then wrap each split in Nifti2DDataset and a PyTorch DataLoader.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


HIPMRI_2D_ROOT = Path("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")


@dataclass(frozen=True)
class NiftiPair:
    """One 2D image, its segmentation mask and its original patient ID."""

    patient_id: str
    image_path: Path
    mask_path: Path


def _find_slices(folder: Path, prefix: str) -> dict:
    """Index case_/seg_ files by their (patient, week, slice) identity."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {folder}")
    pattern = re.compile(rf"{prefix}_([0-9]+)_week_([0-9]+)_slice_([0-9]+)\.nii(?:\.gz)?")
    files = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or not path.name.endswith((".nii", ".nii.gz")):
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected HipMRI filename: {path.name}")
        key = tuple(map(int, match.groups()))
        if key in files:
            raise ValueError(f"Duplicate patient/week/slice {key} in {folder}")
        files[key] = path
    if not files:
        raise ValueError(f"No NIfTI slices found in {folder}")
    return files


def load_hipmri_splits(root: str | Path = HIPMRI_2D_ROOT) -> dict[str, list[NiftiPair]]:
    """Preserve the supplied splits and reject patient overlap between them.

    Match case_004_week_0_slice_0.nii.gz to seg_004_week_0_slice_0.nii.gz.
    Return keys are train, val and test. No files are moved or re-split.
    """
    root = Path(root).expanduser()
    splits = {}
    patient_splits = {}
    for split, folder in (("train", "train"), ("val", "validate"), ("test", "test")):
        images = _find_slices(root / f"keras_slices_{folder}", "case")
        masks = _find_slices(root / f"keras_slices_seg_{folder}", "seg")
        if images.keys() != masks.keys():
            raise ValueError(
                f"Unmatched files in {split}: missing masks for "
                f"{sorted(images.keys() - masks.keys())[:5]}, missing images for "
                f"{sorted(masks.keys() - images.keys())[:5]}"
            )
        splits[split] = []
        for key in sorted(images):
            patient_id = f"{key[0]:03d}"
            previous = patient_splits.get(patient_id)
            if previous is not None and previous != split:
                raise ValueError(f"Patient {patient_id} appears in both {previous} and {split}")
            patient_splits[patient_id] = split
            splits[split].append(NiftiPair(patient_id, images[key], masks[key]))
    return splits


def _read_2d(path: Path) -> tuple:
    """Read a slice; trailing singleton axes are allowed, volumes are rejected."""
    nifti = nib.load(str(path))
    data = nifti.get_fdata(caching="unchanged")
    while data.ndim > 2 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 2 or 0 in data.shape:
        raise ValueError(f"Expected a single 2D slice, got {nifti.shape}: {path}")
    return nifti, data


class Nifti2DDataset(Dataset):
    """Return (image, target) for each paired 2D slice.

    image: float32 [1, H, W], z-score normalized per slice.
    target: int64 [H, W], ready for CrossEntropyLoss.

    class_values lists the audited original label IDs in model-channel order,
    background first. Use the same values for all splits. For example, (0, 5)
    maps original labels 0 and 5 to target indices 0 and 1. One-hot encoding for
    Dice loss belongs in the loss function. No resizing or augmentation is done.
    """

    def __init__(self, pairs: list[NiftiPair], *, class_values: tuple[int, ...]):
        self.pairs = list(pairs)
        self.class_values = tuple(class_values)
        if not self.pairs:
            raise ValueError("The dataset needs at least one image/mask pair.")
        if (not self.class_values or len(set(self.class_values)) != len(self.class_values)
                or any(not isinstance(value, (int, np.integer)) for value in self.class_values)):
            raise ValueError("class_values must contain unique integer label IDs.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[index]
        image_nifti, image = _read_2d(pair.image_path)
        mask_nifti, mask = _read_2d(pair.mask_path)
        if image.shape != mask.shape:
            raise ValueError(f"Image/mask shape mismatch: {pair.image_path}")
        if not np.allclose(image_nifti.affine, mask_nifti.affine, rtol=0, atol=1e-4):
            raise ValueError(f"Image/mask affine mismatch: {pair.image_path}")
        if not np.isfinite(image).all():
            raise ValueError(f"Non-finite image intensities: {pair.image_path}")
        if not np.isfinite(mask).all() or not np.equal(mask, np.rint(mask)).all():
            raise ValueError(f"Mask must contain finite integer labels: {pair.mask_path}")
        unknown = np.setdiff1d(np.unique(mask), self.class_values)
        if unknown.size:
            raise ValueError(f"Unknown mask labels {unknown.tolist()}: {pair.mask_path}")

        target = np.empty(mask.shape, dtype=np.int64)
        for channel, label in enumerate(self.class_values):
            target[mask == label] = channel
        image = (image - image.mean()) / max(float(image.std()), 1e-8)
        return (torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(target))
