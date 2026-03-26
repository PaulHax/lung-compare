"""Download model weights for VoxTell and pretrained nnUNet.

TotalSegmentator and LungTumorMask auto-download weights on first run.
VoxTell and the pretrained nnUNet need manual weight setup.

Usage:
    uv run python scripts/setup_weights.py
    uv run python scripts/setup_weights.py --model voxtell
    uv run python scripts/setup_weights.py --model nnunet_pretrained
"""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def setup_voxtell():
    """Download VoxTell weights from HuggingFace."""
    model_dir = MODELS_DIR / "voxtell"
    weights_dir = model_dir / "weights"

    if weights_dir.exists() and any(weights_dir.iterdir()):
        print("VoxTell weights already downloaded")
        return

    print("Downloading VoxTell weights from HuggingFace...")
    weights_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            str(model_dir / ".venv" / "bin" / "python"), "-c",
            f"""
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="mrokuss/VoxTell",
    local_dir="{weights_dir}",
)
print("Done")
""",
        ],
        check=True,
    )


def setup_nnunet_pretrained():
    """Download pretrained nnUNet weights from Kaggle."""
    model_dir = MODELS_DIR / "nnunet_pretrained"
    weights_dir = model_dir / "weights" / "nnUNet" / "3d_fullres" / "Task998_LungCTSegmentation"

    if weights_dir.exists() and any(weights_dir.iterdir()):
        print("nnUNet pretrained weights already downloaded")
        return

    print("nnUNet pretrained weights must be downloaded manually from Kaggle:")
    print("  https://www.kaggle.com/models/dejankuhn/ct-lung-tumor-segmentation-model")
    print()
    print(f"After downloading, extract into:")
    print(f"  {weights_dir}")
    print()
    print("Expected structure:")
    print(f"  {weights_dir}/nnUNetTrainerV2__nnUNetPlansv2.1/")
    print(f"    fold_0/model_final_checkpoint.model")
    print(f"    fold_0/model_final_checkpoint.model.pkl")
    print(f"    plans.pkl")


SETUP_FUNCS = {
    "voxtell": setup_voxtell,
    "nnunet_pretrained": setup_nnunet_pretrained,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=list(SETUP_FUNCS.keys()),
        help="Set up only this model's weights (default: all)",
    )
    args = parser.parse_args()

    models = [args.model] if args.model else list(SETUP_FUNCS.keys())
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Setting up: {model_name}")
        print(f"{'='*60}")
        SETUP_FUNCS[model_name]()


if __name__ == "__main__":
    main()
