"""Download a small set of NSCLC-Radiomics cases from TCIA.

Downloads CT and RTSTRUCT DICOM series for a handful of patients.
DICOM files land in data/dicom/{patient_id}/{CT,RTSTRUCT}/.

Usage:
    uv run python scripts/download_cases.py
    uv run python scripts/download_cases.py --patients LUNG1-001 LUNG1-002 LUNG1-003
"""

import argparse
import shutil
from pathlib import Path

from tcia_utils import nbia

# Three patients picked to give a range of tumor sizes based on
# manual inspection of the dataset.  Override with --patients.
DEFAULT_PATIENTS = ["LUNG1-001", "LUNG1-008", "LUNG1-014"]

COLLECTION = "NSCLC-Radiomics"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DICOM_DIR = DATA_DIR / "dicom"


def get_series_for_patient(patient_id: str) -> dict[str, str]:
    """Return dict mapping modality -> SeriesInstanceUID for CT and RTSTRUCT."""
    series_list = nbia.getSeries(collection=COLLECTION, patientId=patient_id)
    result = {}
    for s in series_list:
        mod = s["Modality"]
        if mod in ("CT", "RTSTRUCT"):
            result[mod] = s["SeriesInstanceUID"]
    return result


def download_series(series_uid: str, dest: Path) -> None:
    """Download a single DICOM series into dest directory."""
    # tcia_utils downloads into a flat directory; we use a temp location then move
    tmp = dest.parent / f".tmp_{dest.name}"
    tmp.mkdir(parents=True, exist_ok=True)
    nbia.downloadSeries(
        series_data=[series_uid],
        input_type="list",
        path=str(tmp),
    )
    # tcia_utils creates a subfolder named by the series UID
    downloaded = list(tmp.iterdir())
    if not downloaded:
        raise RuntimeError(f"No files downloaded for series {series_uid}")
    # Move to final location
    dest.mkdir(parents=True, exist_ok=True)
    for item in downloaded:
        if item.is_dir():
            for f in item.iterdir():
                shutil.move(str(f), str(dest / f.name))
            item.rmdir()
        else:
            shutil.move(str(item), str(dest / item.name))
    tmp.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patients",
        nargs="+",
        default=DEFAULT_PATIENTS,
        help="Patient IDs to download (default: %(default)s)",
    )
    args = parser.parse_args()

    for patient_id in args.patients:
        print(f"\n{'='*60}")
        print(f"Patient: {patient_id}")
        print(f"{'='*60}")

        series = get_series_for_patient(patient_id)
        if "CT" not in series:
            print(f"  WARNING: No CT series found, skipping")
            continue
        if "RTSTRUCT" not in series:
            print(f"  WARNING: No RTSTRUCT series found, skipping")
            continue

        for modality, uid in series.items():
            dest = DICOM_DIR / patient_id / modality
            if dest.exists() and any(dest.iterdir()):
                print(f"  {modality}: already downloaded, skipping")
                continue
            print(f"  {modality}: downloading {uid[:40]}...")
            download_series(uid, dest)
            n_files = len(list(dest.glob("*.dcm")))
            print(f"  {modality}: {n_files} files")

    print(f"\nDone. DICOM data in: {DICOM_DIR}")


if __name__ == "__main__":
    main()
