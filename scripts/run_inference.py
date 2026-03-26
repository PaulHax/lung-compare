"""Run all 4 segmentation models on the example cases.

Expects NIfTI CTs in data/nifti/{patient_id}/ct.nii.gz.
Outputs predictions to data/predictions/{model_name}/{patient_id}/pred.nii.gz.

Must be run inside a Slurm GPU allocation:
    srun --gres=gpu:1 --mem=64G uv run python scripts/run_inference.py

Or run a single model:
    srun --gres=gpu:1 --mem=64G uv run python scripts/run_inference.py --model totalsegmentator
"""

import argparse
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
NIFTI_DIR = DATA_DIR / "nifti"
PRED_DIR = DATA_DIR / "predictions"
MODELS_DIR = ROOT / "models"


def get_patient_cases() -> list[tuple[str, Path]]:
    """Return list of (patient_id, ct_path) for all available cases."""
    cases = []
    for patient_dir in sorted(NIFTI_DIR.iterdir()):
        ct = patient_dir / "ct.nii.gz"
        if ct.exists():
            cases.append((patient_dir.name, ct))
    return cases


def run_totalsegmentator(cases: list[tuple[str, Path]]) -> None:
    """Run TotalSegmentator lung_nodules task."""
    model_name = "totalsegmentator"
    model_dir = MODELS_DIR / "totalsegmentator"
    print(f"\n{'='*60}")
    print(f"Model: TotalSegmentator (lung_nodules)")
    print(f"{'='*60}")

    for patient_id, ct_path in cases:
        out_dir = PRED_DIR / model_name / patient_id
        pred_path = out_dir / "pred.nii.gz"
        if pred_path.exists():
            print(f"  {patient_id}: already done, skipping")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        # TotalSegmentator outputs individual label files into a directory;
        # we'll use --ml flag to get a single multilabel file instead.
        ts_out = out_dir / "ts_raw"
        ts_out.mkdir(exist_ok=True)

        print(f"  {patient_id}: running...")
        t0 = time.time()
        subprocess.run(
            [
                str(model_dir / ".venv" / "bin" / "TotalSegmentator"),
                "-i", str(ct_path),
                "-o", str(ts_out),
                "-ta", "lung_nodules",
                "--ml",
            ],
            check=True,
        )
        dt = time.time() - t0

        # The multilabel output is ts_raw.nii.gz (next to the dir)
        ml_file = out_dir / "ts_raw.nii.gz"
        if ml_file.exists():
            ml_file.rename(pred_path)
        # Also check inside the dir for individual label files
        nodule_files = list(ts_out.glob("lung_nodule*"))
        if nodule_files and not pred_path.exists():
            import shutil
            shutil.copy2(str(nodule_files[0]), str(pred_path))

        print(f"  {patient_id}: done ({dt:.1f}s)")


def run_lungtumormask(cases: list[tuple[str, Path]]) -> None:
    """Run LungTumorMask."""
    model_name = "lungtumormask"
    model_dir = MODELS_DIR / "lungtumormask"
    print(f"\n{'='*60}")
    print(f"Model: LungTumorMask")
    print(f"{'='*60}")

    for patient_id, ct_path in cases:
        out_dir = PRED_DIR / model_name / patient_id
        pred_path = out_dir / "pred.nii.gz"
        if pred_path.exists():
            print(f"  {patient_id}: already done, skipping")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  {patient_id}: running...")
        t0 = time.time()
        subprocess.run(
            [
                str(model_dir / ".venv" / "bin" / "lungtumormask"),
                str(ct_path),
                str(pred_path),
                "--lung-filter",
                "--cpu",  # torch<=1.11 lacks A6000 (sm_86) CUDA kernels
            ],
            check=True,
        )
        dt = time.time() - t0
        print(f"  {patient_id}: done ({dt:.1f}s)")


def run_voxtell(cases: list[tuple[str, Path]]) -> None:
    """Run VoxTell with text prompt."""
    model_name = "voxtell"
    model_dir = MODELS_DIR / "voxtell"
    print(f"\n{'='*60}")
    print(f"Model: VoxTell (prompt='lung tumor')")
    print(f"{'='*60}")

    for patient_id, ct_path in cases:
        out_dir = PRED_DIR / model_name / patient_id
        pred_path = out_dir / "pred.nii.gz"
        if pred_path.exists():
            print(f"  {patient_id}: already done, skipping")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        weights_dir = model_dir / "weights" / "voxtell_v1.1"
        if not weights_dir.exists():
            print(f"  ERROR: VoxTell weights not found. Run: uv run python scripts/setup_weights.py --model voxtell")
            return

        print(f"  {patient_id}: running...")
        t0 = time.time()
        subprocess.run(
            [
                str(model_dir / ".venv" / "bin" / "voxtell-predict"),
                "-i", str(ct_path),
                "-o", str(out_dir),
                "-m", str(weights_dir),
                "-p", "lung tumor",
            ],
            check=True,
        )
        dt = time.time() - t0

        # VoxTell output filename may vary — find it and rename
        nii_files = [f for f in out_dir.glob("*.nii.gz") if f.name != "pred.nii.gz"]
        if nii_files:
            nii_files[0].rename(pred_path)

        print(f"  {patient_id}: done ({dt:.1f}s)")


def run_nnunet_pretrained(cases: list[tuple[str, Path]]) -> None:
    """Run pretrained nnUNet v1 lung CT model."""
    model_name = "nnunet_pretrained"
    model_dir = MODELS_DIR / "nnunet_pretrained"
    print(f"\n{'='*60}")
    print(f"Model: Pretrained nnUNet (CT)")
    print(f"{'='*60}")

    # nnUNet v1 expects a folder of input files named like case_0000.nii.gz
    for patient_id, ct_path in cases:
        out_dir = PRED_DIR / model_name / patient_id
        pred_path = out_dir / "pred.nii.gz"
        if pred_path.exists():
            print(f"  {patient_id}: already done, skipping")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        # Create temp input folder with nnUNet naming convention
        input_dir = out_dir / "input"
        input_dir.mkdir(exist_ok=True)
        link_path = input_dir / f"{patient_id}_0000.nii.gz"
        if not link_path.exists():
            link_path.symlink_to(ct_path)

        print(f"  {patient_id}: running...")
        t0 = time.time()

        env = {
            "RESULTS_FOLDER": str(model_dir / "weights"),
        }

        subprocess.run(
            [
                str(model_dir / ".venv" / "bin" / "nnUNet_predict"),
                "-i", str(input_dir),
                "-o", str(out_dir),
                "-tr", "nnUNetTrainerV2",
                "-m", "3d_fullres",
                "-p", "nnUNetPlansv2.1",
                "-t", "Task998_LungCTSegmentation",
            ],
            check=True,
            env={**__import__("os").environ, **env},
        )
        dt = time.time() - t0

        # Rename nnUNet output
        nnunet_out = out_dir / f"{patient_id}.nii.gz"
        if nnunet_out.exists():
            nnunet_out.rename(pred_path)

        # Clean up temp input dir
        link_path.unlink(missing_ok=True)
        input_dir.rmdir()

        print(f"  {patient_id}: done ({dt:.1f}s)")


def run_voxtell_nodule(cases: list[tuple[str, Path]]) -> None:
    """Run VoxTell with 'lung nodule' prompt."""
    model_name = "voxtell_nodule"
    model_dir = MODELS_DIR / "voxtell"
    print(f"\n{'='*60}")
    print(f"Model: VoxTell (prompt='lung nodule')")
    print(f"{'='*60}")

    for patient_id, ct_path in cases:
        out_dir = PRED_DIR / model_name / patient_id
        pred_path = out_dir / "pred.nii.gz"
        if pred_path.exists():
            print(f"  {patient_id}: already done, skipping")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        weights_dir = model_dir / "weights" / "voxtell_v1.1"
        if not weights_dir.exists():
            print(f"  ERROR: VoxTell weights not found. Run: uv run python scripts/setup_weights.py --model voxtell")
            return

        print(f"  {patient_id}: running...")
        t0 = time.time()
        subprocess.run(
            [
                str(model_dir / ".venv" / "bin" / "voxtell-predict"),
                "-i", str(ct_path),
                "-o", str(out_dir),
                "-m", str(weights_dir),
                "-p", "lung nodule",
            ],
            check=True,
        )
        dt = time.time() - t0

        # VoxTell output filename may vary — find it and rename
        nii_files = [f for f in out_dir.glob("*.nii.gz") if f.name != "pred.nii.gz"]
        if nii_files:
            nii_files[0].rename(pred_path)

        print(f"  {patient_id}: done ({dt:.1f}s)")


MODEL_RUNNERS = {
    "totalsegmentator": run_totalsegmentator,
    "lungtumormask": run_lungtumormask,
    "voxtell": run_voxtell,
    "voxtell_nodule": run_voxtell_nodule,
    "nnunet_pretrained": run_nnunet_pretrained,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=list(MODEL_RUNNERS.keys()),
        help="Run only this model (default: all)",
    )
    args = parser.parse_args()

    cases = get_patient_cases()
    if not cases:
        print(f"No cases found in {NIFTI_DIR}. Run download_cases.py + preprocess.py first.")
        return

    print(f"Cases: {[c[0] for c in cases]}")

    models = [args.model] if args.model else list(MODEL_RUNNERS.keys())
    for model_name in models:
        try:
            MODEL_RUNNERS[model_name](cases)
        except Exception as e:
            print(f"\n  ERROR running {model_name}: {e}")
            print(f"  Continuing with next model...\n")


if __name__ == "__main__":
    main()
