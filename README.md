# MOSAIC

**Connecting histopathology, clinical information, and genomics for outcome prediction.**

[Model weights](https://huggingface.co/Astaxanthin/MOSAIC) · [KEEP patch encoder](https://huggingface.co/Astaxanthin/KEEP) · [Quick start](#quick-start) · [Reproduction](#reproduce-the-example-tasks)

MOSAIC combines separately pretrained biomedical representations to predict **overall survival (OS)** and **progression-free survival (PFS)**. It supports clinical, genomic, and WSI inputs, including missing modalities, and exposes shared WSI embeddings for downstream prediction tasks.

This repository provides a minimal Hugging Face inference interface and two reproducible examples:

| Example | Inputs | Task |
|---|---|---|
| **Yale HER2** | Shared WSI embeddings | Five-fold treatment-response prediction with a linear probe |
| **HCMI** | Clinical data, pathology, TP53 SNV | OS prediction across four matched input settings |

## Quick start

### 1. Install

Python 3.10 or newer is recommended. Install PyTorch for your CPU/CUDA environment, then:

```bash
git clone https://github.com/MAGIC-AI4Med/MOSAIC.git
cd MOSAIC
pip install -r requirements.txt
```

### 2. Load MOSAIC

```python
import h5py
import torch
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(
    "Astaxanthin/MOSAIC", trust_remote_code=True,
).to(device).eval()
```

A local model directory can replace the Hub ID. `Astaxanthin/KEEP` is the upstream patch encoder; **load `Astaxanthin/MOSAIC` for the calls below**. The MOSAIC tokenizer is bundled and loaded automatically.

### 3. Predict risks and extract a WSI embedding

```python
# Features and coordinates must describe the same patches in the same order.
with h5py.File("slide_features.h5", "r") as f:
    features = f["features"][:]  # [N, 768], KEEP patch features
    coords = f["coords"][:]      # [N, 2], integer spatial coordinates

output = model.predict(
    patch_features=features,
    patch_coords=coords,
    cancer_type="Lung Adenocarcinoma",
    clinical={"age": 65, "sex": "Male", "stage": "II"},
    genomics={"snv": ["TP53"], "cnv": ["CDKN2A"]},
)

print("WSI embedding:", output["wsi_features"].shape)  # [1, 768]
print("OS risk:", output["os_risk"].item())
print("PFS risk:", output["pfs_risk"].item())

# Clinical only: no WSI or genomics required.
clinical_only = model.predict(
    cancer_type="Lung Adenocarcinoma",
    clinical={"age": 65, "sex": "Male", "stage": "II"},
)

# Vision only: omit all text inputs, including cancer_type.
vision_only = model.predict(patch_features=features, patch_coords=coords)

# Clinical + genomics: leave CNV unknown when only SNV was evaluated.
clinical_genomics = model.predict(
    cancer_type="Lung Adenocarcinoma",
    clinical={"age": 65, "sex": "Male", "stage": "II"},
    genomics={"snv": ["TP53"], "cnv": -1},
)
```

## Which representation should I use?

| Representation | Shape / location | Purpose |
|---|---|---|
| **Patch features** | Input `patch_features`, `[N, 768]` | Local histopathology representations extracted with KEEP. These are the input to the WSI aggregator, not patient-level predictions. |
| **Shared WSI embedding** | `wsi_features`, `[1, 768]` | A slide-level representation before endpoint-specific adaptors. Use it as input to a downstream classifier, such as the Yale HER2 treatment-response probe below. Other tasks require their own labels, training and validation. |
| **Per-view WSI embeddings** | `view_wsi_features`, `[V, 768]` | Spatial-view representations for inspecting or reproducing slide-level feature aggregation. |
| **OS / PFS risk** | `os_risk`, `pfs_risk` | Endpoint-specific predictions from the available modalities. Use for survival discrimination, e.g. C-index; higher values indicate greater predicted risk. |
| **Per-view risks** | `view_os_risk`, `view_pfs_risk`, `[V]` | Endpoint-specific predictions before independent max pooling over views. |

The shared WSI embedding is a **visual representation**, even when clinical/genomic inputs are supplied. It is not a fused multimodal embedding. The language representations and endpoint-specific adapted features are used internally and are not separate outputs of the basic API.

**Treatment response and survival use different prediction paths:** the treatment example trains a logistic-regression probe on WSI embeddings; it does not treat the OS/PFS risk scores as treatment-response probabilities. No treatment-response classifier is bundled in the survival checkpoint.

### Spatial and patient-level aggregation

`predict` handles spatial padding and deterministic coverage with 80 × 80 patch-grid views. It independently takes the maximum OS and PFS risk across views. The shared WSI embedding is pooled by an elementwise maximum; survival risks are not recomputed from that pooled embedding. Use `predict_view` only for an already prepared view.

Multiple slides are **not** combined automatically. These examples reproduce the original patient aggregation:

- **Treatment:** max over view embeddings → mean over slide embeddings → L2 normalization → linear probe.
- **Survival:** max over view risks → mean over slide risks → patient-level C-index.

For treatment reproduction, view batching also matters. The original evaluator batches up to **8 views**; the public `predict` loop evaluates views individually. Nyström attention's approximate inverse includes batch-wide normalization, so changing this can change embeddings. The Yale script uses the **HF-loaded backbone with the original view batching** by default. The HCMI script uses `predict` with the validated survival settings.

## Input conventions

- Extract patch features with the same KEEP encoder and preprocessing used for training. A matching feature dimension alone does not make another encoder interchangeable.
- Supply matching integer `(x, y)` coordinates in the feature-extraction coordinate system, not arbitrary patch indices.
- Clinical keys: `age`, `sex`, `stage`, `grade`, `subtype`. Aliases `diagnosis age`, `gender`, and `tumor stage` are accepted; `cancer_type` is supplied separately.
- Genomic `snv` / `cnv` values are lists of qualifying gene names, not raw variants or numeric copy-number values. For a panel, filter the observed list to that panel first.
- Omitted genomic fields, `None`, or `-1` mean **unknown**. `[]` means **known absence of qualifying alterations in the evaluated gene set**. Do not replace missing measurements with empty lists.
- Each call handles one slide/patient input. Outputs are CPU tensors. Without WSI, `wsi_features` is `None`, and per-view outputs are absent.

## Reproduce the example tasks

Use either the **interactive terminal scripts** or the **step-by-step notebooks**:

| Task | Python script | Notebook |
|---|---|---|
| Yale HER2 treatment response | [reproduce_yale_her2.py](examples/reproduce_yale_her2.py) | [Yale HER2 walkthrough](notebooks/01_yale_her2_treatment.ipynb) |
| HCMI four-setting OS | [reproduce_hcmi_os.py](examples/reproduce_hcmi_os.py) | [HCMI walkthrough](notebooks/02_hcmi_os.ipynb) |

Running a script without arguments in a terminal prompts for paths and settings. Press Enter to retain defaults. All paths are configurable; no workstation-specific paths are hard-coded. Model weights, patient metadata and patch features are supplied separately and are not included in this code package.

Suggested layout:

```text
MOSAIC/
├── examples/
├── notebooks/
├── data/
│   ├── yale_her2/
│   │   ├── yale_her2_dataset.json
│   │   └── features/*.h5
│   └── hcmi/
│       ├── hcmi_dataset.json
│       └── features/*.h5
└── outputs/                     # Generated locally; ignored by Git
```

For notebooks:

```bash
pip install -r requirements-notebooks.txt
jupyter lab
```

### Yale HER2 — treatment-response prediction

```bash
python examples/reproduce_yale_her2.py
```

Noninteractive equivalent:

```bash
python examples/reproduce_yale_her2.py --no-interactive --features data/yale_her2/features --metadata data/yale_her2/yale_her2_dataset.json --output outputs/yale_her2
```

The script checks the validated cohort of **85 patients (36 responders, 49 non-responders)**, sorts case IDs, and reproduces five-fold stratified cross-validation with seed 42. A cohort fingerprint prevents silently evaluating a different set of patients. StandardScaler and a balanced logistic-regression classifier are fitted using training folds only. Held-out probabilities use a 0.5 classification threshold.

Optionally pass `--reference path/to/original_predictions.json` to check saved folds and compare each patient's new and original prediction. The original all-dataset JSON and its extracted Yale result are both accepted. A reference file is not required to reproduce the deterministic split.

| Metric | Original evaluation | Verified HF reproduction* |
|---|---:|---:|
| Mean five-fold AUROC | 0.761587 | 0.761587 |
| Fold AUROC standard deviation | 0.124025 | 0.124025 |
| Mean balanced accuracy | 0.674921 | 0.674921 |
| Mean accuracy | 0.682353 | 0.682353 |
| Pooled out-of-fold AUROC | 0.755669 | 0.755102 |

*All five individual-fold metrics and all 85 predicted classes matched. Probabilities were not bitwise identical; the mean absolute difference was 0.001545. Pooled OOF AUROC uses comparisons across folds and is distinct from mean fold AUROC. Runtime versions and view batching can affect reproduction; the script records configuration and model fingerprints.*

Outputs include `frozen_folds.csv`, `patient_embeddings.npz`, `oof_predictions.csv`, `fold_metrics.csv`, `metrics.json`, and a resumable slide-embedding cache.

### HCMI — OS with four input settings

```bash
python examples/reproduce_hcmi_os.py
```

Noninteractive equivalent:

```bash
python examples/reproduce_hcmi_os.py --no-interactive --features data/hcmi/features --data data/hcmi/hcmi_dataset.json --bootstrap 1000 --output outputs/hcmi_os
```

All settings use the same feature-available, OS-valid test patients. The genomic input is **TP53 SNV only**; CNV remains unknown (`-1`). These are continuous-risk predictions: no gene-direction weights or high/low-risk grouping enters the risk or C-index computation.

| Setting | Inputs | Verified OS C-index |
|---|---|---:|
| C | Clinical | 0.675214 |
| C + P | Clinical + pathology | 0.700855 |
| C + G | Clinical + TP53 SNV | 0.689459 |
| C + P + G | All three | 0.723647 |

The verified cohort contains **35 patients, 69 slides and 15 OS events**. The script saves patient and slide risks, Harrell C-index, percentile 95% CIs from 1,000 paired patient-level bootstrap samples, and differences from clinical-only performance. These figures are reproduction checks on this cohort, not guarantees for other data.

Outputs include `patient_predictions.csv`, `slide_predictions.csv`, `metrics.csv`, `run_config.json`, and a resumable slide-risk cache. Excluded records and reasons are written when applicable.

### Metadata structure

Metadata JSON is keyed by slide ID. Each slide maps to `<slide stem>.h5`, containing `features` and `coords`. The following entries are **schema examples**, not real patient records:

```json
{
  "example_slide.svs": {
    "case_id": "example_patient",
    "project": "pancan_hcmi_2025|Pancreatic_Cancer",
    "state": "test",
    "cancer type": "Pancreatic Cancer",
    "clinical": {"diagnosis age": 65, "gender": "Male", "tumor stage": "II"},
    "surv_prior": {"grade": "G2", "subtype": "Pancreatic Adenocarcinoma", "mutation": ["TP53"], "cnv": -1},
    "survival": {"os_state": 1, "os_time": 730}
  }
}
```

Yale metadata requires `case_id` and a treatment-response label. The evaluator reads top-level `treatment_response` / `raw_treatment_response`, then nested `task.treatment_response` / `surv_prior.treatment_response`. Recognized positive labels are `effective`, `responder`, `pcr`; negative labels are `invalid`, `nonresponder`, `rd`. **In this dataset convention, `invalid` means non-response, not a missing label.**

### Reproducibility options

- `--model /path/to/MOSAIC`: use local weights instead of the Hub.
- `--revision COMMIT_SHA`: pin a Hub snapshot for repeatable downloads.
- `--device cpu` or `--device cuda`: choose the inference device.
- `--validate-only`: check cohort/metadata (and Yale folds) without model inference.
- `--help`: list all arguments, including the Yale `predict` comparison mode.

Caches are reused only when their recorded model/input configuration matches. Use a new output directory after changing model, device or evaluation settings. For HCMI, `--limit` is a smoke-test option only; its metrics are explicitly marked as such.

## Scope and acknowledgments

MOSAIC risk scores are relative predictions, not calibrated survival probabilities. WSI embeddings require a separately fitted and validated predictor for each downstream task. This release supports research reproduction and does not establish clinical utility.

We acknowledge [KEEP](https://huggingface.co/Astaxanthin/KEEP) for the patch representation and tokenizer/configuration assets. The KEEP image encoder is not bundled with the MOSAIC survival checkpoint.

<!-- Before publishing: add the paper citation, author/contact details and an
explicit project license. No license is inferred from dependency licenses. -->
