## Feasibility Review: 3D Improved U-Net for Prostate MRI Segmentation (HipMRI 3D)

**Project:** 2.2 Medical Image Segmentation | **Model:** 3D Improved U-Net (Hard) | **Author:** Einar Fugle

### 1. User Need, Scope & Acceptance Criteria
**Intended user:** radiotherapy planners who manually contour the prostate and nearby organs (bladder, rectum) on pelvic MRI.
**Prototype purpose:** produce automatic draft contours that a clinician reviews and corrects, and find out whether full 3D context removes the hazardous errors a 2D model makes at the prostate base and apex slices and at the thin rectum/bladder interfaces, at an acceptable compute cost.

**Acceptance criteria (testable on the held-out, patient-level test split):**
1. Dice ≥ 0.7 for every class on the test set, with per-class Dice and IoU reported.
2. Prostate Dice on base and apex slices is no worse than the 2D U-Net baseline (target: a measurable improvement).
3. Fewer false-positive voxels leaking into the rectal wall and bladder than the 2D baseline (measured from the difference overlays).
4. Peak training VRAM ≤ [X] GB, so it fits one Rangpur GPU (e.g. A100) at batch size 1–2.
5. Inference time ≤ [Y] s per volume, and total training ≤ [Z] GPU-hours.

### 2. Model Choice & Course Concepts
- **Data:** downsampled HipMRI 3D (38 patients, 211 Nifti volumes). The split is strictly by patient (for example about 70/15/15), so no patient appears in more than one set. Intensity is z-score normalised per volume and labels are one-hot encoded; images and masks keep exactly the same spatial alignment.
- **Model:** 3D Improved U-Net (Isensee et al. [2]). It uses residual context modules with dropout, a localisation path with upsampling, instance normalisation and leaky ReLU, and deep supervision through segmentation outputs added up from several decoder levels.
- **Loss and training:** Dice loss plus cross-entropy to handle the class imbalance (the prostate is small compared with the background), the Adam optimiser with learning-rate decay, and 3D augmentation (flips, small rotations, intensity changes).
- **Baseline:** a 2D U-Net trained on axial slices from the same split, for a direct comparison of 2D and 3D.
- **Course concepts:** convolutional encoder-decoders, skip connections, residual learning, normalisation, overlap-based losses for imbalanced data, and generalisation through patient-level validation.

### 3. Preliminary Feasibility Evidence
*(TODO: fill in with dataset details and baseline measurments)*
- **Data audit:** [number of volumes per split], [volume shape and voxel spacing], [class voxel shares, e.g. background xx %, prostate xx %], and confirmation that the image and mask headers match.
- **Smoke test:** the 2D baseline trained for [N] epochs on a subset reached prostate Dice [a.bc]. A forward and backward pass of the 3D model on one volume worked, with peak VRAM [X] GB and [t] s per epoch on [GPU].

### 4. Risks, Budget & Fallback
| Risk | Mitigation |
|---|---|
| GPU memory too small for 3D volumes | batch size 1, mixed precision (AMP), patch-based training |
| Poor accuracy on base and apex slices, or on small classes | Dice + CE loss, class weighting, deep supervision, evaluation per slice region |
| Data leakage between patients | split by patient ID, checked in code with an assertion |
| Rangpur queue or time limits | checkpoint every epoch so training can resume, short smoke runs locally |

- **Compute budget:** about [20–40] GPU-hours on Rangpur in total (smoke tests, about 3 full training runs, the baseline and evaluation).
- **Next experiment:** train the full 3D model for 50 epochs and compare it with the 2D baseline per class and per slice region.
- **Fallback:** if Improved U-Net does not fit in memory or does not converge, a standard 3D U-Net with fewer filters, or patch-based training with sliding-window inference.