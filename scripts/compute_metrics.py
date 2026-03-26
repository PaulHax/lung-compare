"""Compute segmentation metrics for all models against ground truth.

For each model and patient, computes:
  - Dice Similarity Coefficient (DSC)
  - Intersection over Union (IoU)
  - Hausdorff Distance 95% (HD95) in mm
  - Average Surface Distance (ASD) in mm
  - Detection: whether the prediction overlaps ground truth at all
  - Volume of prediction in cm³

Outputs a summary table to stdout and saves CSV to data/metrics.csv.

Usage:
    uv run python scripts/compute_metrics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
NIFTI_DIR = DATA_DIR / "nifti"
PRED_DIR = DATA_DIR / "predictions"


# For multilabel predictions, which label to use as the target.
# TotalSegmentator lung_nodules outputs: 1=lung, 2=nodules.
# For most models, any nonzero voxel is the prediction.
MODEL_TARGET_LABEL = {
    "totalsegmentator": 2,  # label 2 = nodules (label 1 = lung parenchyma)
}


def load_binary_mask(path: Path, model_name: str = "") -> tuple[np.ndarray, tuple[float, ...]]:
    """Load a NIfTI file and return (binary array, voxel spacing in mm)."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()  # (x, y, z) in mm
    target_label = MODEL_TARGET_LABEL.get(model_name)
    if target_label is not None:
        return (arr == target_label).astype(np.uint8), spacing
    # Default: anything > 0 is foreground
    return (arr > 0).astype(np.uint8), spacing


def compute_surface_distances(gt: np.ndarray, pred: np.ndarray, spacing: tuple[float, ...]):
    """Compute surface distances using the surface-distance package."""
    import surface_distance

    # surface_distance expects (z, y, x) spacing but SimpleITK gives (x, y, z)
    # GetArrayFromImage returns (z, y, x), spacing is (x, y, z)
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    return surface_distance.compute_surface_distances(
        gt.astype(bool), pred.astype(bool), spacing_mm=spacing_zyx
    )


def compute_case_metrics(gt_path: Path, pred_path: Path, model_name: str = "") -> dict:
    """Compute all metrics for a single case."""
    gt, spacing = load_binary_mask(gt_path)
    pred, _ = load_binary_mask(pred_path, model_name=model_name)

    # Handle shape mismatch (some models may resample)
    if gt.shape != pred.shape:
        # Resample pred to gt space
        gt_img = sitk.ReadImage(str(gt_path))
        pred_img = sitk.ReadImage(str(pred_path))
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(gt_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        pred_img = resampler.Execute(pred_img)
        pred = (sitk.GetArrayFromImage(pred_img) > 0).astype(np.uint8)

    gt_sum = gt.sum()
    pred_sum = pred.sum()
    intersection = (gt & pred).sum()
    union = (gt | pred).sum()

    # Volume in cm³
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    gt_vol_cc = gt_sum * voxel_vol_mm3 / 1000.0
    pred_vol_cc = pred_sum * voxel_vol_mm3 / 1000.0

    result = {
        "gt_vol_cc": round(gt_vol_cc, 1),
        "pred_vol_cc": round(pred_vol_cc, 1),
        "detected": bool(intersection > 0) if gt_sum > 0 else None,
    }

    # Dice and IoU
    if gt_sum == 0 and pred_sum == 0:
        result["dice"] = 1.0
        result["iou"] = 1.0
    elif gt_sum == 0 or pred_sum == 0:
        result["dice"] = 0.0
        result["iou"] = 0.0
    else:
        result["dice"] = round(2.0 * intersection / (gt_sum + pred_sum), 4)
        result["iou"] = round(intersection / union, 4) if union > 0 else 0.0

    # Surface distances (only if both masks are non-empty)
    if gt_sum > 0 and pred_sum > 0:
        try:
            import surface_distance

            surf = compute_surface_distances(gt, pred, spacing)
            hd95 = surface_distance.compute_robust_hausdorff(surf, 95)
            asd = (
                surface_distance.compute_average_surface_distance(surf)
            )
            # asd returns (gt_to_pred, pred_to_gt) — average both
            result["hd95_mm"] = round(hd95, 2)
            result["asd_mm"] = round((asd[0] + asd[1]) / 2, 2)
        except Exception as e:
            result["hd95_mm"] = None
            result["asd_mm"] = None
            result["surface_error"] = str(e)
    else:
        result["hd95_mm"] = None
        result["asd_mm"] = None

    return result


def main():
    # Discover all models and patients
    model_dirs = sorted(PRED_DIR.iterdir()) if PRED_DIR.exists() else []
    patient_dirs = sorted(NIFTI_DIR.iterdir()) if NIFTI_DIR.exists() else []

    if not model_dirs or not patient_dirs:
        print("No predictions or ground truth found. Run inference first.")
        return

    rows = []
    for model_dir in model_dirs:
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for patient_dir in patient_dirs:
            if not patient_dir.is_dir():
                continue
            patient_id = patient_dir.name

            gt_path = patient_dir / "gt_tumor.nii.gz"
            pred_path = model_dir / patient_id / "pred.nii.gz"

            if not gt_path.exists() or not pred_path.exists():
                continue

            print(f"  {model_name} / {patient_id}...", end=" ", flush=True)
            metrics = compute_case_metrics(gt_path, pred_path, model_name=model_name)
            metrics["model"] = model_name
            metrics["patient"] = patient_id
            rows.append(metrics)
            print(f"DSC={metrics['dice']}")

    df = pd.DataFrame(rows)

    # Reorder columns
    col_order = [
        "model", "patient", "dice", "iou", "hd95_mm", "asd_mm",
        "detected", "gt_vol_cc", "pred_vol_cc",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    # Print per-case results
    print(f"\n{'='*80}")
    print("Per-case results")
    print(f"{'='*80}")
    print(df.to_string(index=False))

    # Print summary by model
    print(f"\n{'='*80}")
    print("Summary by model (mean ± std)")
    print(f"{'='*80}")
    summary = df.groupby("model").agg(
        dice_mean=("dice", "mean"),
        dice_std=("dice", "std"),
        iou_mean=("iou", "mean"),
        hd95_mean=("hd95_mm", "mean"),
        asd_mean=("asd_mm", "mean"),
        detection_rate=("detected", "mean"),
        pred_vol_mean=("pred_vol_cc", "mean"),
    ).round(3)
    print(summary.to_string())

    # Save CSV
    csv_path = DATA_DIR / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
