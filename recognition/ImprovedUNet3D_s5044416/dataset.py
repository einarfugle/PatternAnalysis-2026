"""HipMRI data loading and patient-level splitting for the 2D vs 3D U-Net comparison.

Both models read the same 3D scans:
1. discover_scans() pairs each semantic_MRs/<patient>_Week<n>_LFOV.nii.gz volume with
   the label volume of the same patient and week in semantic_labels_only.
2. make_patient_split() assigns whole patients, never single scans, to train/val/test.
   The split is saved once to patient_split.json and reused by every model.
3. load_scan() reorients a scan to RAS and lays it out as [D, H, W], where D is the
   axial (inferior to superior) slice index. It downsamples every axis by
   DEFAULT_DOWNSAMPLE = 2, so 256 x 256 x 128 voxels become 128 x 128 x 64 (block mean
   for MR, majority vote for labels), then z-scores the MR using only that scan's own
   voxels, so no statistics are shared between splits.
4. VolumeDataset serves whole scans to the 3D model. AxialSliceDataset serves the axial
   slices of the same preprocessed scans to the 2D baseline.

Create the split once on Rangpur with `python dataset.py --write-split`; running
`python dataset.py` checks the saved split against the data.
"""

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


DEFAULT_ROOT = "/home/groups/comp3710/HipMRI_Study_open"
IMAGE_DIR = "semantic_MRs"
LABEL_DIR = "semantic_labels_only"
SPLIT_FILE = Path(__file__).resolve().parent / "patient_split.json"
SPLIT_NAMES = ("train", "val", "test")
# Label IDs in the order given in the HipMRI study description.
LABEL_NAMES = ("background", "body", "bone", "bladder", "rectum", "prostate")
# Integer factor applied to every axis; 1 keeps the native ~1.6 mm voxel grid.
DEFAULT_DOWNSAMPLE = 2

_SCAN_NAME = re.compile(r"(?P<patient>.+?)_Week(?P<week>[0-9]+)(?:_.+)?\.nii(?:\.gz)?")


@dataclass(frozen=True)
class Scan:
    """One weekly MR volume, its label volume and the patient it belongs to."""

    patient_id: str
    week: int
    image_path: Path
    label_path: Path


def _index_volumes(folder: Path) -> dict:
    """Index a folder's NIfTI volumes by (patient, week)."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {folder}")
    volumes = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or not path.name.endswith((".nii", ".nii.gz")):
            continue
        match = _SCAN_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected HipMRI filename: {path.name}")
        key = (match["patient"], int(match["week"]))
        if key in volumes:
            raise ValueError(f"Two volumes for patient {key[0]} week {key[1]} in {folder}: "
                             f"{volumes[key].name}, {path.name}")
        volumes[key] = path
    if not volumes:
        raise ValueError(f"No NIfTI volumes found in {folder}")
    return volumes


def discover_scans(root: str | Path = DEFAULT_ROOT, image_dir: str = IMAGE_DIR,
                   label_dir: str = LABEL_DIR) -> list[Scan]:
    """Pair every MR volume with its label volume by (patient, week).

    The patient ID is the filename text before _Week, e.g. B006 in
    B006_Week0_LFOV.nii.gz. Unmatched or duplicated scans raise an error.
    """
    root = Path(root).expanduser()
    images = _index_volumes(root / image_dir)
    labels = _index_volumes(root / label_dir)
    if images.keys() != labels.keys():
        raise ValueError(
            f"Unmatched scans: missing labels for {sorted(images.keys() - labels.keys())[:5]}, "
            f"missing images for {sorted(labels.keys() - images.keys())[:5]}"
        )
    return [Scan(patient, week, images[patient, week], labels[patient, week])
            for patient, week in sorted(images)]


def _proportional(total: int, sizes: list[int]) -> list[int]:
    """Share total places between groups in proportion to their sizes (largest remainder)."""
    quotas = [total * size / sum(sizes) for size in sizes]
    counts = [int(quota) for quota in quotas]
    by_remainder = sorted(range(len(sizes)), key=lambda i: quotas[i] - counts[i], reverse=True)
    for i in by_remainder[:total - sum(counts)]:
        counts[i] += 1
    return counts


def make_patient_split(scans: list[Scan], *, val_patients: int = 5, test_patients: int = 7,
                       seed: int = 42) -> dict[str, list[str]]:
    """Randomly assign whole patients to train/val/test with a fixed seed.

    Single-scan patients and patients with a weekly series are shuffled separately,
    and the val/test places are shared between the two groups in proportion to size.
    """
    counts = Counter(scan.patient_id for scan in scans)
    if val_patients < 1 or test_patients < 1 or val_patients + test_patients >= len(counts):
        raise ValueError("Need at least one val patient, one test patient and one training patient.")
    groups = [sorted(p for p, n in counts.items() if n == 1),
              sorted(p for p, n in counts.items() if n > 1)]
    sizes = [len(group) for group in groups]
    rng = random.Random(seed)
    split = {name: [] for name in SPLIT_NAMES}
    for group, n_test, n_val in zip(groups, _proportional(test_patients, sizes),
                                    _proportional(val_patients, sizes)):
        if n_test + n_val > len(group):
            raise ValueError("Too many val/test patients for the single-scan/series groups.")
        rng.shuffle(group)
        split["test"] += group[:n_test]
        split["val"] += group[n_test:n_test + n_val]
        split["train"] += group[n_test + n_val:]
    return {name: sorted(patients) for name, patients in split.items()}


def check_split(split: dict, patients: set[str]) -> None:
    """Require every patient in exactly one non-empty split, with no unknown IDs."""
    owner = {}
    for name in SPLIT_NAMES:
        if not split.get(name):
            raise ValueError(f"Split '{name}' is missing or empty.")
        for patient in split[name]:
            if patient in owner:
                raise ValueError(f"Patient {patient} appears in both {owner[patient]} and {name}")
            owner[patient] = name
    unknown, missing = sorted(owner.keys() - patients), sorted(patients - owner.keys())
    if unknown or missing:
        raise ValueError(f"Split does not match the data (made from another copy?): unknown "
                         f"patients {unknown[:5]}, unassigned patients {missing[:5]}")


def split_scans(scans: list[Scan], split: dict) -> dict[str, list[Scan]]:
    """Group scans by split after checking that the split covers exactly these patients."""
    check_split(split, {scan.patient_id for scan in scans})
    owner = {patient: name for name in SPLIT_NAMES for patient in split[name]}
    grouped = {name: [] for name in SPLIT_NAMES}
    for scan in scans:
        grouped[owner[scan.patient_id]].append(scan)
    return grouped


def save_split(split: dict, path: str | Path = SPLIT_FILE, note: str = "") -> None:
    """Write the split as JSON. Never overwrite, so the test set cannot silently change."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists. Delete it first to make a new split.")
    data = ({"note": note} if note else {}) | {name: list(split[name]) for name in SPLIT_NAMES}
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_split(path: str | Path = SPLIT_FILE) -> dict[str, list[str]]:
    """Read a split written by save_split()."""
    data = json.loads(Path(path).read_text())
    missing = [name for name in SPLIT_NAMES if name not in data]
    if missing:
        raise ValueError(f"{path} has no {missing} patient list.")
    return {name: list(data[name]) for name in SPLIT_NAMES}


def _downsample_labels(target: torch.Tensor, size: tuple) -> torch.Tensor:
    """Majority vote over each block.

    A tie goes to the label of the voxel that nearest-exact sampling picks from the
    block. Always favouring one label ID would shrink the organs that lose ties.
    """
    sampled = F.interpolate(target[None, None].float(), size=size, mode="nearest-exact")[0, 0]
    best_share = torch.full(size, -1.0)
    result = torch.zeros(size, dtype=torch.int64)
    for label in range(len(LABEL_NAMES)):
        share = F.interpolate((target == label).float()[None, None], size=size, mode="area")[0, 0]
        # The tie-break bonus is far smaller than any difference between two block shares.
        share = share + 1e-6 * (sampled == label)
        better = share > best_share
        result[better] = label
        best_share = torch.where(better, share, best_share)
    return result


def load_scan(scan: Scan,
              downsample: int = DEFAULT_DOWNSAMPLE) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    """Read, reorient, downsample and normalize one scan.

    Returns the MR as float32 [1, D, H, W], labels as int64 [D, H, W] and the voxel
    spacing in mm as (D, H, W). Axes run inferior->superior (D, the axial slice index),
    posterior->anterior (H) and left->right (W). Use downsample=1 for the native grid.
    """
    if type(downsample) is not int or downsample < 1:
        raise ValueError("downsample must be a positive integer.")
    image_nifti = nib.squeeze_image(nib.load(str(scan.image_path)))
    label_nifti = nib.squeeze_image(nib.load(str(scan.label_path)))
    if len(image_nifti.shape) != 3 or image_nifti.shape != label_nifti.shape:
        raise ValueError(f"Expected matching 3D image/label volumes, got {image_nifti.shape} "
                         f"and {label_nifti.shape}: {scan.image_path.name}")
    if not np.allclose(image_nifti.affine, label_nifti.affine, rtol=0, atol=1e-3):
        raise ValueError(f"Image/label affine mismatch: {scan.image_path.name}")
    # Axis flips/permutations only, no resampling: RAS (x, y, z) -> [D, H, W] = (z, y, x).
    image_nifti = nib.as_closest_canonical(image_nifti)
    label_nifti = nib.as_closest_canonical(label_nifti)
    spacing = tuple(float(size) for size in np.linalg.norm(image_nifti.affine[:3, :3], axis=0)[::-1])
    image = image_nifti.get_fdata(dtype=np.float32).transpose(2, 1, 0)
    label = np.asanyarray(label_nifti.dataobj).transpose(2, 1, 0)
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite MR intensities: {scan.image_path.name}")
    if label.dtype.kind not in "iu" and not (np.isfinite(label).all()
                                             and np.equal(label, np.rint(label)).all()):
        raise ValueError(f"Labels must be finite integers: {scan.label_path.name}")
    label = np.ascontiguousarray(label, dtype=np.int64)
    unknown = np.setdiff1d(np.unique(label), np.arange(len(LABEL_NAMES)))
    if unknown.size:
        raise ValueError(f"Unknown labels {unknown.tolist()}: {scan.label_path.name}")

    image = torch.from_numpy(np.ascontiguousarray(image))[None]
    target = torch.from_numpy(label)
    if downsample > 1:
        size = tuple(max(1, n // downsample) for n in target.shape)
        spacing = tuple(s * n / m for s, n, m in zip(spacing, target.shape, size))
        image = F.interpolate(image[None], size=size, mode="area")[0]
        target = _downsample_labels(target, size)
    image = (image - image.mean()) / max(float(image.std(correction=0)), 1e-8)
    return image, target, spacing


class VolumeDataset(Dataset):
    """Whole preprocessed scans for the 3D model: image [1, D, H, W], target [D, H, W].

    Scans can differ in size, so use batch_size=1 (or crop to a common size).
    """

    def __init__(self, scans: list[Scan], *, downsample: int = DEFAULT_DOWNSAMPLE):
        if not scans:
            raise ValueError("The dataset needs at least one scan.")
        self.scans = list(scans)
        self.downsample = downsample

    def __len__(self) -> int:
        return len(self.scans)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, target, _ = load_scan(self.scans[index], self.downsample)
        return image, target


class AxialSliceDataset(Dataset):
    """Axial slices of the same preprocessed scans for the 2D baseline.

    Returns image [1, H, W] and target [H, W]. Every scan is loaded once through
    load_scan() and kept in memory (about 5 MB per scan at the default downsampling,
    40 MB at native resolution), so each slice is normalized exactly as in VolumeDataset.
    """

    def __init__(self, scans: list[Scan], *, downsample: int = DEFAULT_DOWNSAMPLE):
        if not scans:
            raise ValueError("The dataset needs at least one scan.")
        self.scans = list(scans)
        self.images, self.targets, self.slices = [], [], []
        for number, scan in enumerate(self.scans):
            image, target, _ = load_scan(scan, downsample)
            self.images.append(image[0])
            self.targets.append(target.to(torch.uint8))
            self.slices.extend((number, depth) for depth in range(target.shape[0]))

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        number, depth = self.slices[index]
        return self.images[number][depth].unsqueeze(0).clone(), self.targets[number][depth].long()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or check the patient-level HipMRI split.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--label-dir", default=LABEL_DIR)
    parser.add_argument("--split-file", type=Path, default=SPLIT_FILE)
    parser.add_argument("--write-split", action="store_true",
                        help="Create the split file first; an existing file is never overwritten.")
    parser.add_argument("--val-patients", type=int, default=5)
    parser.add_argument("--test-patients", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scans = discover_scans(args.root, args.image_dir, args.label_dir)
    if args.write_split:
        split = make_patient_split(scans, val_patients=args.val_patients,
                                   test_patients=args.test_patients, seed=args.seed)
        save_split(split, args.split_file,
                   note=f"Patient-level split, seed {args.seed}; single-scan and weekly-series "
                        "patients shuffled separately.")
        print(f"Wrote {args.split_file}")
    split = load_split(args.split_file)
    for name, items in split_scans(scans, split).items():
        print(f"{name}: {len(split[name])} patients, {len(items)} scans: {', '.join(split[name])}")
    volumes = [nib.load(str(scan.image_path)) for scan in scans]
    print("MR shapes:", dict(Counter(volume.shape for volume in volumes)))
    print("Voxel sizes (mm):", dict(Counter(
        tuple(round(float(size), 2) for size in volume.header.get_zooms()[:3]) for volume in volumes)))
    print("Orientations:", dict(Counter("".join(nib.aff2axcodes(volume.affine)) for volume in volumes)))


if __name__ == "__main__":
    main()
