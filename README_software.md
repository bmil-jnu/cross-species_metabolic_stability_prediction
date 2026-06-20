# MTMM — Cross-species Microsomal Stability: Prediction & Explanation Software

Standalone software for applying and explaining the multi-task multi-modal (MTMM) model
for liver microsomal metabolic stability in **human (HLM)**, **rat (RLM)**, and
**mouse (MLM)**. It lets researchers run predictions and inspect model explanations
without retraining and without deep computational setup.

> Binary label convention: `0 = stable`, `1 = unstable` (half-life cutoff t½ ≤ 30 min).

---

## Contents

```
model.py                        # MTMM model definition (needed to load the bundle)
dataset_scaffold_modelready.py  # SMILES -> graph / fingerprint / descriptor preprocessing
predict.py                      # prediction CLI (mtmm-predict)
explain.py                      # descriptor SHAP + EdgeSHAPer explanations
export_release_bundle.py        # bundle trained model + scaler + thresholds -> mtmm_release.pt
notebooks/MTMM_predict_colab.ipynb  # browser-based prediction, no local install
examples/README.md              # input CSV format
pyproject.toml  requirements.txt  LICENSE  .gitignore
```

A single release bundle `mtmm_release.pt` (model + descriptor scaler / imputation /
column schema + per-species thresholds) is required at run time; distribute it as a
GitHub Release asset and download it as shown below.

---

## Quick start

### Option A — Google Colab (no local install)

Open `notebooks/MTMM_predict_colab.ipynb` in Colab and run top to bottom. It installs
dependencies in the browser, downloads the bundle, takes a SMILES + SwissADME CSV, and
returns per-species predictions.

[Open in Colab](https://colab.research.google.com/github/bmil-jnu/cross-species_metabolic_stability_prediction/blob/main/notebooks/MTMM_predict_colab.ipynb)

### Option B — Command line

```bash
pip install -e .
mtmm-predict --bundle mtmm_release.pt --input my_compounds.csv --output predictions.csv
```

Output columns: `prob_*` (probability of unstable) and `pred_*` (thresholded label) for
human / rat / mouse.

### Model explanations

```bash
pip install -e ".[explain]"
python explain.py --bundle mtmm_release.pt --input my_compounds.csv --species human --mode both
```

`explain.py` produces descriptor-level SHAP plots (Method S5) and per-bond EdgeSHAPer
attributions drawn on the molecule (Method S6).

---

## Building the release bundle

`predict.py` and `explain.py` need a bundle pairing the trained model with the descriptor
scaler, imputation values, column schema, and per-species thresholds. Build it once with
`export_release_bundle.export_bundle(...)` from the training environment (see that file's
header for arguments).

## Input format

A CSV with a SMILES column (`Cano_Smile`, `SMILES`, or `PUBCHEM_EXT_DATASOURCE_SMILES`)
plus SwissADME descriptor columns. See `examples/README.md`. SwissADME
(http://www.swissadme.ch/) is a web tool; obtain its descriptor CSV there and merge it
with your SMILES before running the tools.

## Installation

Confirmed environment: Python 3.12.5, PyTorch 2.2.0, Torch_Geometric 2.6.1,
scikit-learn 1.6.1. See `requirements.txt`. `torch`, `torch_geometric`, and `rdkit` may
need environment-specific wheels (CUDA / PyG matching).

## Notes

This package is the standalone-software / accessibility deliverable: it provides
prediction and explanation visualization without substantial computational expertise. A
hosted web server / public API is left as future work.

## License / contacts

MIT (see `LICENSE`; fill copyright line). Contacts: munsu931122@jnu.ac.kr, syyoo@jnu.ac.kr
