"""Convert downloaded DICOM cases to NIfTI.

For each patient in data/dicom/{patient_id}/:
  - CT DICOM → data/nifti/{patient_id}/ct.nii.gz
  - RTSTRUCT  → data/nifti/{patient_id}/gt_tumor.nii.gz  (GTV labelmap)

Optionally deletes DICOM after conversion to save disk space (--cleanup).

Usage:
    uv run python scripts/preprocess.py
    uv run python scripts/preprocess.py --cleanup
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from rt_utils import RTStructBuilder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DICOM_DIR = DATA_DIR / "dicom"
NIFTI_DIR = DATA_DIR / "nifti"

# Common GTV ROI name patterns in NSCLC-Radiomics (case-insensitive match)
GTV_PATTERNS = ["gtv", "gross tumor", "tumor", "primary"]


def convert_ct(dicom_dir: Path, output_path: Path) -> sitk.Image:
    """Convert CT DICOM series to NIfTI using dcm2niix, return as SimpleITK image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            ["dcm2niix", "-z", "y", "-f", "ct", "-o", tmpdir, str(dicom_dir)],
            check=True,
            capture_output=True,
        )
        nii_files = list(Path(tmpdir).glob("ct*.nii.gz"))
        if not nii_files:
            raise RuntimeError(f"dcm2niix produced no output for {dicom_dir}")
        # Take the first (usually only) output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(nii_files[0]), str(output_path))

    return sitk.ReadImage(str(output_path))


def find_gtv_roi(roi_names: list[str]) -> str | None:
    """Find the GTV/tumor ROI name from a list of ROI names."""
    for name in roi_names:
        lower = name.lower()
        for pattern in GTV_PATTERNS:
            if pattern in lower:
                return name
    return None


def convert_rtstruct(
    rtstruct_dir: Path, ct_dicom_dir: Path, ct_image: sitk.Image, output_path: Path
) -> None:
    """Convert RTSTRUCT to a binary NIfTI labelmap aligned to the CT."""
    # Find the RTSTRUCT file
    rs_files = list(rtstruct_dir.glob("*.dcm"))
    if not rs_files:
        raise RuntimeError(f"No DICOM files in {rtstruct_dir}")

    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=str(ct_dicom_dir),
        rt_struct_path=str(rs_files[0]),
    )

    roi_names = rtstruct.get_roi_names()
    print(f"    ROI names: {roi_names}")

    gtv_name = find_gtv_roi(roi_names)
    if gtv_name is None:
        print(f"    WARNING: No GTV-like ROI found in {roi_names}")
        print(f"    Using first ROI: {roi_names[0]}")
        gtv_name = roi_names[0]

    print(f"    Using ROI: '{gtv_name}'")
    mask_3d = rtstruct.get_roi_mask_by_name(gtv_name)  # (H, W, D) bool array

    # rt-utils returns (row, col, slice) matching the CT DICOM geometry.
    # Convert to SimpleITK image with same metadata as CT.
    mask_arr = mask_3d.astype(np.uint8)
    # rt-utils mask is (H, W, num_slices) — need to transpose for SimpleITK
    # SimpleITK expects (x, y, z) indexing via GetImageFromArray which takes (z, y, x)
    mask_sitk = sitk.GetImageFromArray(mask_arr.transpose(2, 0, 1))
    mask_sitk.SetOrigin(ct_image.GetOrigin())
    mask_sitk.SetSpacing(ct_image.GetSpacing())
    mask_sitk.SetDirection(ct_image.GetDirection())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(mask_sitk, str(output_path))

    n_voxels = int(mask_arr.sum())
    spacing = ct_image.GetSpacing()
    vol_cc = n_voxels * spacing[0] * spacing[1] * spacing[2] / 1000.0
    print(f"    Tumor voxels: {n_voxels}, volume: {vol_cc:.1f} cm³")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete DICOM files after conversion to save disk space",
    )
    args = parser.parse_args()

    patient_dirs = sorted(DICOM_DIR.iterdir()) if DICOM_DIR.exists() else []
    if not patient_dirs:
        print(f"No patients found in {DICOM_DIR}. Run download_cases.py first.")
        return

    for patient_dir in patient_dirs:
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        ct_dir = patient_dir / "CT"
        rs_dir = patient_dir / "RTSTRUCT"

        if not ct_dir.exists() or not rs_dir.exists():
            print(f"Skipping {patient_id}: missing CT or RTSTRUCT")
            continue

        ct_out = NIFTI_DIR / patient_id / "ct.nii.gz"
        gt_out = NIFTI_DIR / patient_id / "gt_tumor.nii.gz"

        print(f"\n{'='*60}")
        print(f"Patient: {patient_id}")
        print(f"{'='*60}")

        # Convert CT
        if ct_out.exists():
            print(f"  CT: already converted")
            ct_image = sitk.ReadImage(str(ct_out))
        else:
            print(f"  CT: converting DICOM → NIfTI...")
            ct_image = convert_ct(ct_dir, ct_out)
            print(f"  CT: {ct_out}")

        # Convert RTSTRUCT
        if gt_out.exists():
            print(f"  GT: already converted")
        else:
            print(f"  GT: converting RTSTRUCT → labelmap...")
            convert_rtstruct(rs_dir, ct_dir, ct_image, gt_out)
            print(f"  GT: {gt_out}")

        # Cleanup DICOM
        if args.cleanup:
            print(f"  Cleaning up DICOM...")
            shutil.rmtree(patient_dir)

    print(f"\nDone. NIfTI data in: {NIFTI_DIR}")


if __name__ == "__main__":
    main()
