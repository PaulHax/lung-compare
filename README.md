# Lung Tumor Segmentation Model Comparison

Evaluating off-the-shelf segmentation models for fully automatic lung cancer tumor segmentation on CT.

## Models

| Model | Type |
|-------|------|
| VoxTell | Vision-language (MaskFormer + Qwen3) |
| Pretrained nnUNet (CT) | 3D full-res nnUNet v1 |
| LungTumorMask | Teacher-student U-Net |
| TotalSegmentator (lung_nodules) | nnUNet-based |

## Evaluation Dataset

3 cases from **NSCLC-Radiomics** (TCIA) with expert-delineated GTV contours:

| Patient | Tumor Volume |
|---------|-------------|
| LUNG1-001 | 162.0 cm³ (large) |
| LUNG1-008 | 43.1 cm³ (medium) |
| LUNG1-014 | 21.7 cm³ (small) |

## Results

### Per-case DSC

| Model | LUNG1-001 | LUNG1-008 | LUNG1-014 |
|-------|-----------|-----------|-----------|
| VoxTell (nodule prompt) | 0.53 | 0.00 | 0.21 |
| VoxTell (tumor prompt) | 0.44 | 0.00 | 0.28 |
| nnUNet pretrained | 0.53 | 0.00 | 0.00 |
| LungTumorMask | 0.43 | 0.00 | 0.18 |
| TotalSegmentator | 0.38 | 0.00 | 0.00 |

### Summary

| Model | Mean DSC | Mean HD95 (mm) | Mean ASD (mm) | Detection Rate |
|-------|----------|----------------|---------------|----------------|
| VoxTell (nodule) | 0.246 | 40.7 | 23.0 | 67% |
| VoxTell (tumor) | 0.239 | 76.7 | 32.2 | 67% |
| LungTumorMask | 0.206 | 95.6 | 35.1 | 67% |
| nnUNet pretrained | 0.177 | 100.3 | 51.9 | 33% |
| TotalSegmentator | 0.126 | 92.0 | 54.9 | 33% |

**No model meets the "usable as first pass" threshold** (DSC >= 0.65, HD95 <= 15mm, detection >= 90%).

LUNG1-008 was missed by every model (except TotalSegmentator which over-segments). This suggests custom nnUNet training is needed.

## Usage

```bash
# Download 3 example cases from TCIA
uv run python scripts/download_cases.py

# Convert DICOM to NIfTI (CT + ground truth labelmap)
uv run python scripts/preprocess.py
uv run python scripts/preprocess.py --cleanup  # delete DICOM after

# Install model environments
cd models/totalsegmentator && uv sync && cd ../..
cd models/lungtumormask && uv sync && cd ../..
cd models/voxtell && uv sync && cd ../..
cd models/nnunet_pretrained && uv sync && cd ../..

# Download model weights (VoxTell + nnUNet pretrained)
uv run python scripts/setup_weights.py

# Run inference (requires GPU via Slurm)
srun --gres=gpu:1 --mem=64G uv run python scripts/run_inference.py

# Compute metrics
uv run python scripts/compute_metrics.py
```
