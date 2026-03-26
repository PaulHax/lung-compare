# Lung Tumor Segmentation Model Evaluation Plan

## Goal

Evaluate off-the-shelf segmentation models for fully automatic lung cancer tumor segmentation on CT, with no per-case user interaction. Identify which model (if any) is suitable as a first-pass segmenter for commercial clinical trials. If no off-the-shelf model meets quality requirements, train a custom nnUNet model.

## Context

- Commercial company, clinical trial use case
- Models must have commercial-friendly licenses or be self-trained
- Fully automatic inference (no per-case user interaction)

## Hardware

- NVIDIA A6000 (48 GB VRAM)

## Models Under Evaluation

| #   | Model                               | Type                                 | Prompt               | License                                    | Commercial?            | Source                                                                                                             |
| --- | ----------------------------------- | ------------------------------------ | -------------------- | ------------------------------------------ | ---------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 1   | **VoxTell**                         | Vision-language (MaskFormer + Qwen3) | Text: `"lung tumor"` | Code: Apache 2.0, Weights: CC-BY-NC-SA 4.0 | **No** (research only) | [github.com/MIC-DKFZ/VoxTell](https://github.com/MIC-DKFZ/VoxTell)                                                 |
| 2   | **Pretrained nnUNet (CT)**          | 3D full-res nnUNet v1                | None                 | No license specified                       | **Unknown**            | [github.com/MonCarFa/PET-CT-Lung-Segmentation-Models](https://github.com/MonCarFa/PET-CT-Lung-Segmentation-Models) |
| 3   | **LungTumorMask**                   | Teacher-student U-Net                | None                 | MIT                                        | **Yes**                | [github.com/VemundFredriksen/LungTumorMask](https://github.com/VemundFredriksen/LungTumorMask)                     |
| 4   | **TotalSegmentator** (lung_nodules) | nnUNet-based                         | None                 | Open (contact for some subtasks)           | **Yes**                | [github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)                               |

> **Note:** Models 1 and 2 are included for research benchmarking only. Only models with confirmed commercial licenses (3, 4, and custom-trained nnUNet below) are candidates for clinical trial deployment.

## Evaluation Dataset

**Primary: NSCLC-Radiomics (Lung1)**

- 304 NSCLC patients with expert-delineated tumor contours
- Publicly available on [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/)
- Large enough for meaningful statistics, diverse tumor sizes

**Secondary (optional): MSD Task06 Lung**

- 63 NSCLC CTs with ground truth masks
- Smaller but well-established benchmark
- [HuggingFace](https://huggingface.co/datasets/Novel-BioMedAI/Medical_Segmentation_Decathlon) or [medicaldecathlon.com](http://medicaldecathlon.com/)

## Metrics

### Overlap

| Metric                                | What it measures                                        |
| ------------------------------------- | ------------------------------------------------------- |
| **Dice Similarity Coefficient (DSC)** | Voxel-level overlap between prediction and ground truth |
| **Intersection over Union (IoU)**     | Stricter overlap measure                                |

### Surface Distance

| Metric                             | What it measures                            |
| ---------------------------------- | ------------------------------------------- |
| **Hausdorff Distance 95% (HD95)**  | Worst-case boundary error (95th percentile) |
| **Average Surface Distance (ASD)** | Mean boundary deviation                     |

### Detection

| Metric                   | What it measures                                     |
| ------------------------ | ---------------------------------------------------- |
| **Sensitivity (Recall)** | Fraction of true tumors detected (any overlap)       |
| **False positive rate**  | Predicted regions with no corresponding ground truth |

### Practical

| Metric                        | What it measures            |
| ----------------------------- | --------------------------- |
| **Inference time per volume** | Wall-clock seconds, GPU     |
| **Peak GPU memory**           | VRAM usage during inference |

## Methodology

### Phase 1: Setup and Preprocessing

1. Download NSCLC-Radiomics dataset from TCIA
2. Convert DICOM to NIfTI (use `dcm2niix` or similar)
3. Align ground truth RT-STRUCT contours to the CT volumes as binary masks
4. Verify alignment by visual spot-check on 5-10 cases
5. Install all three models in isolated conda/venv environments

### Phase 2: Inference

Run each model on the full dataset with no manual intervention.

```bash
# VoxTell (research evaluation only — weights are non-commercial)
voxtell-predict -i $CASE -o $OUT/voxtell/ -m $MODEL_PATH -p "lung tumor"

# Pretrained nnUNet (research evaluation — no license on weights)
nnUNet_predict -i $INPUT_FOLDER -o $OUT/nnunet/ \
  -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes \
  -m 3d_fullres -p nnUNetPlansv2.1 -t Task998_LungCTSegmentation

# LungTumorMask (MIT — commercial OK)
lungtumormask $CASE $OUT/lungtumormask/output.nii.gz

# TotalSegmentator lung_nodules (commercial OK)
TotalSegmentator -i $CASE -o $OUT/totalseg/ -ta lung_nodules
```

Record inference time and peak GPU memory for each run.

### Phase 3: Compute Metrics

- Use a consistent evaluation script for all models (e.g., `surface-distance` and `medpy` Python packages, or write a single script with SimpleITK)
- Compute per-case DSC, IoU, HD95, ASD
- Flag detection: prediction mask > 0 overlapping ground truth mask > 0
- Compute per-case and aggregate (mean, median, std) statistics

### Phase 4: Stratified Analysis

Break down results by:

- **Tumor volume**: small (< 10 cm^3), medium (10-100 cm^3), large (> 100 cm^3)
- **Tumor location**: central vs peripheral
- **Failure modes**: missed tumors, gross over-segmentation, fragmented predictions

### Phase 5: Qualitative Review

- Visually inspect the 10 best and 10 worst cases (by DSC) for each model
- Look for systematic errors: chest wall inclusion, mediastinal bleed-through, atelectasis confusion
- Document representative good/bad examples with screenshots

## VoxTell Prompt Ablation (Bonus)

Since VoxTell is text-prompted, test multiple prompts on a 20-case subset:

| Prompt                         | Rationale             |
| ------------------------------ | --------------------- |
| `"lung tumor"`                 | Generic               |
| `"lung cancer"`                | Disease-level         |
| `"non-small cell lung cancer"` | Histology-specific    |
| `"pulmonary nodule"`           | Nodule framing        |
| `"lung mass"`                  | Radiology terminology |

Pick the best-performing prompt for the full evaluation.

## Success Criteria

| Tier                                  | DSC     | HD95     | Detection Rate |
| ------------------------------------- | ------- | -------- | -------------- |
| Usable as first pass (edit from here) | >= 0.65 | <= 15 mm | >= 90%         |
| Good (minimal corrections needed)     | >= 0.75 | <= 10 mm | >= 95%         |
| Excellent                             | >= 0.85 | <= 5 mm  | >= 98%         |

## Expected Timeline (Off-the-Shelf Evaluation)

| Step                                      | Estimate |
| ----------------------------------------- | -------- |
| Dataset download and preprocessing        | 1-2 days |
| Model installation and environment setup  | 1 day    |
| Inference (all 4 models, ~300 cases each) | 1 day    |
| Metric computation and analysis           | 1 day    |
| Qualitative review and writeup            | 1 day    |

---

## Phase 2: Custom nnUNet Training

If no off-the-shelf model meets quality requirements, or to establish the best achievable performance for commercial deployment.

### Why nnUNet

- Framework is Apache 2.0 — you own the trained weights, fully commercial
- Self-configuring: automatically determines preprocessing, architecture, and hyperparameters
- Consistently top-performing on medical segmentation benchmarks
- Fully automatic at inference (no prompts)

### Training Data

| Dataset         | Cases | Source                                                                   |
| --------------- | ----- | ------------------------------------------------------------------------ |
| MSD Task06 Lung | 63    | [medicaldecathlon.com](http://medicaldecathlon.com/)                     |
| NSCLC-Radiomics | 304   | [TCIA](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) |
| **Combined**    | ~367  | Merge both for maximum training data                                     |

> Check each dataset's license for restrictions on derivative model distribution. MSD is CC-BY-SA 4.0. NSCLC-Radiomics uses TCIA Data Usage Policy.

### Data Preparation

1. Convert all data to nnUNet expected format (`nnUNet_raw/DatasetXXX_LungTumor/`)
2. Images as `_0000.nii.gz`, labels as `.nii.gz` in `imagesTr`/`labelsTr`
3. Create `dataset.json` with channel and label definitions
4. Run nnUNet preprocessing:
   ```bash
   nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
   ```

### Training Strategy

**Quick validation (start here):**

Train a single fold of 3d_fullres to validate the pipeline works and get an early quality signal.

```bash
nnUNetv2_train DATASET_ID 3d_fullres 0
```

**Full training (if single fold looks good):**

Train all 5 folds for ensembling and proper cross-validation.

```bash
nnUNetv2_train DATASET_ID 3d_fullres 0
nnUNetv2_train DATASET_ID 3d_fullres 1
nnUNetv2_train DATASET_ID 3d_fullres 2
nnUNetv2_train DATASET_ID 3d_fullres 3
nnUNetv2_train DATASET_ID 3d_fullres 4
```

**Optional:** Also train 3d_lowres and 3d_cascade_fullres, then let nnUNet pick the best configuration:

```bash
nnUNetv2_find_best_configuration DATASET_ID -c 3d_fullres 3d_lowres 3d_cascade_fullres
```

### Estimated Training Time (NVIDIA A6000, 48 GB)

| Configuration | Dataset         | Per Fold  | 5-Fold Total |
| ------------- | --------------- | --------- | ------------ |
| 3d_fullres    | MSD Task06 (63) | ~12-24 h  | ~3-5 days    |
| 3d_fullres    | Combined (~367) | ~3-5 days | ~2-3 weeks   |
| 3d_lowres     | Combined (~367) | ~1-2 days | ~5-10 days   |

> The A6000's 48 GB allows larger patch sizes than typical GPUs, which can improve convergence. nnUNet will auto-configure this.

### Inference with Trained Model

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER \
  -d DATASET_ID -c 3d_fullres -f 0  # single fold
  # or omit -f to use all 5 folds (ensemble, best quality)
```

### Training Timeline

| Step                                     | Estimate  |
| ---------------------------------------- | --------- |
| Data prep (DICOM/NIfTI to nnUNet format) | 1-2 days  |
| nnUNet preprocessing                     | ~1 hour   |
| Train single fold (3d_fullres, Task06)   | ~1 day    |
| Evaluate single fold, decide next step   | 0.5 days  |
| Full 5-fold on combined data (if needed) | 2-3 weeks |
| Final evaluation against off-the-shelf   | 1 day     |

### Decision Gate

After single-fold training on MSD Task06:

- **If DSC > 0.70:** Promising. Proceed with combined dataset and full 5-fold.
- **If DSC 0.55-0.70:** Marginal. Try combined dataset single fold before committing to full training.
- **If DSC < 0.55:** Investigate data quality, preprocessing, or consider alternative approaches.

---

## Tools and Libraries

- `dcm2niix` -- DICOM to NIfTI conversion
- `rt-utils` or `dcmrtstruct2nii` -- RT-STRUCT to NIfTI mask conversion
- `SimpleITK` or `medpy` -- metric computation
- `matplotlib` / `napari` -- visualization and qualitative review
- `pandas` -- tabulation and statistical summary
- `nnunetv2` -- model training framework (Apache 2.0)
