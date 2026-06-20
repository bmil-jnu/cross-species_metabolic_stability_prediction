# Cross-species liver microsomal metabolic stability prediction

Authors: Subhin Seomun, Sunyong Yoo

A species-aware, multi-task, multi-modal framework that predicts liver microsomal
metabolic stability for **human (HLM)**, **rat (RLM)**, and **mouse (MLM)**. The model
fuses SMILES-derived fingerprints, a graph neural network, and in-silico ADME /
physicochemical descriptors, and explains its decisions with descriptor-level SHAP and
EdgeSHAPer plus fragment–ADME enrichment.

> Binary label convention: `0 = stable`, `1 = unstable` (half-life cutoff t½ ≤ 30 min).

---

## Use the model without deep setup

The fastest ways to apply the model to your own compounds:

### Option A — Google Colab (no local install)

Open `notebooks/MTMM_predict_colab.ipynb` in Colab and run top to bottom. It installs
dependencies in the browser, downloads the model bundle, takes a SMILES + SwissADME CSV,
and returns per-species predictions:

[Open in Colab](https://colab.research.google.com/github/bmil-jnu/cross-species_metabolic_stability_prediction/blob/main/notebooks/MTMM_predict_colab.ipynb)

### Option B — Command line

```bash
pip install -e .
mtmm-predict --bundle mtmm_release.pt --input my_compounds.csv --output predictions.csv
```

Input: a CSV with a SMILES column (`Cano_Smile`, `SMILES`, or
`PUBCHEM_EXT_DATASOURCE_SMILES`) plus SwissADME descriptor columns (see
`examples/README.md`). Output: `prob_*` and `pred_*` columns for human / rat / mouse.

### Model explanations

```bash
pip install -e ".[explain]"
python explain.py --bundle mtmm_release.pt --input my_compounds.csv --species human --mode both
```

`explain.py` produces descriptor-level SHAP plots (Method S5) and per-bond EdgeSHAPer
attributions drawn on the molecule (Method S6).

---

## Repository structure

```
.
├── model.py                       # MTMM model (graph + fingerprint + descriptor fusion)
├── dataset_scaffold_modelready.py # data prep, scaffold K-fold loaders, fingerprints, descriptors
├── utile.py  train.py  Focal_loss.py
├── main.ipynb                     # training: scaffold-disjoint K-fold CV from one full CSV
├── predict.py                     # inference CLI (mtmm-predict)
├── export_release_bundle.py       # bundle trained model + scaler + thresholds -> mtmm_release.pt
├── explain.py                     # SHAP (S5) + EdgeSHAPer (S6)
├── notebooks/MTMM_predict_colab.ipynb
├── analysis/                      # SHAP / EdgeSHAPer / Heatmap analysis (revision version)
├── examples/                      # example input format
├── data/                          # dataset location (see data/README.md)
├── pyproject.toml  requirements.txt  LICENSE
```

---

## Training

`main.ipynb` runs scaffold-disjoint 10-fold cross-validation from a single full CSV and
writes fold-wise metrics, a mean±SD summary, and out-of-fold predictions. Set the data
path in the final cell (`DATA_PATH`, `DATASET_NAME`).

## Building the release bundle

`predict.py` and `explain.py` need a single bundle that pairs the trained model with the
descriptor scaler, imputation values, column schema, and per-species thresholds. Build it
once with `export_release_bundle.export_bundle(...)` inside the training notebook (see the
file header for arguments). Distribute the bundle as a GitHub Release asset.

## Installation

Confirmed environment (paper): Python 3.12.5, PyTorch 2.2.0, Torch_Geometric 2.6.1,
scikit-learn 1.6.1. See `requirements.txt`. Note that `torch`, `torch_geometric`, and
`rdkit` may require environment-specific installation (CUDA / PyG wheels).

## Data availability

The dataset is not committed here. <!-- TODO: deposit on Zenodo and add the DOI below. -->
Dataset DOI: `TODO`. Training set derived from PubChem BioAssay; external validation from
ChEMBL (CC-BY-SA — attribute accordingly).

## Citation

<!-- TODO: add citation once published. -->
`TODO`

## Contacts

- munsu931122@jnu.ac.kr
- syyoo@jnu.ac.kr

## License

MIT (see `LICENSE`). <!-- TODO: confirm license choice and fill the copyright line. -->
