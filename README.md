# Cross-species liver microsomal metabolic stability prediction

Code, trained model, data, and tools for the paper
**"Cross-species multi-task learning with molecular and ADME descriptors for liver microsomal metabolic stability"**
(Subhin Seomun, Sunyong Yoo).

The model predicts binary liver microsomal metabolic stability for **human (HLM)**,
**rat (RLM)**, and **mouse (MLM)** — `unstable` if half-life t½ ≤ 30 min, `stable` if
t½ > 30 min (label convention: `0 = stable`, `1 = unstable`).

---

## Overview

Liver microsomal stability is an important early filter in lead optimization, but
cross-species prediction is difficult because metabolic pathways differ between species
and most models offer limited mechanistic insight. This work proposes a **cross-species,
multi-task** framework that combines three complementary molecular modalities:

- **SMILES-derived fingerprints** (Morgan, MACCS, RDKit),
- **molecular graphs** (atom/bond features through a graph neural network encoder), and
- **in-silico ADME / physicochemical descriptors**.

These modalities are fused and passed to a **shared network plus species-specific
output networks**, so the model captures general structure–metabolism patterns while
still expressing species-dependent effects. It is trained on a curated set of nearly
19,000 microsomal half-life measurements from **PubChem BioAssay** (HLM, RLM, MLM) and
evaluated with **stratified 10-fold Bemis–Murcko scaffold cross-validation** and species-specific thresholds. Across species the model outperforms
conventional machine-learning and single-task deep-learning baselines.

For interpretability, the framework provides **descriptor-level SHAP** attribution
(which ADME / physicochemical features drive a prediction), **EdgeSHAPer** per-bond
attribution (which substructures stabilize or destabilize a molecule), and a
**fragment–ADME enrichment** analysis linking local motifs to whole-molecule properties.

---

## Try it without local setup (Google Colab)

Open the notebook in Colab and run top to bottom — it installs dependencies in the
browser, downloads the released model, takes your input file, and returns predictions and
explanation figures:

[Colab File](https://github.com/bmil-jnu/cross-species_metabolic_stability_prediction/tree/main/Notebooks/MTMM_predict_colab.ipynb)

---

## Repository structure

```
.
├── model.py                       # model definition (graph + fingerprint + descriptor fusion)
├── dataset_scaffold_modelready.py # data prep, scaffold K-fold loaders, fingerprints, descriptors
├── utile.py  train.py  Focal_loss.py
├── main.ipynb                     # training: scaffold-disjoint 10-fold cross-validation
├── predict.py                     # inference CLI (mtmm-predict)
├── export_release_bundle.py       # pack trained model + scaler + thresholds -> mtmm_release.pt
├── explain.py                     # descriptor SHAP + EdgeSHAPer
├── notebooks/MTMM_predict_colab.ipynb
├── analysis/                      # interpretability analysis (SHAP / EdgeSHAPer / fragment enrichment)
├── examples/                      # input format
├── data/                          # datasets
```

## Installation

Tested with Python 3.12.5, PyTorch 2.2.0, Torch_Geometric 2.6.1, scikit-learn 1.6.1
(see `requirements.txt`). From the repository root:

```bash
pip install -e .
```

Note: `torch`, `torch_geometric`, and `rdkit` may require environment-specific
installation (CUDA / PyG wheels).

## Usage

### Predict

```bash
mtmm-predict --bundle mtmm_release.pt --input my_compounds.csv --output predictions.csv
```

The input CSV needs a SMILES column (`Cano_Smile`, `SMILES`, or
`PUBCHEM_EXT_DATASOURCE_SMILES`) plus SwissADME descriptor columns; output columns are
`prob_*` (probability of unstable) and `pred_*` (thresholded label) for human / rat / mouse.
See `examples/` for the expected input format.

> The descriptor modality depends on SwissADME-derived features. Obtain the descriptor
> table from the free SwissADME web tool (http://www.swissadme.ch/) and merge it with your
> SMILES before running the tools.

### Explain

```bash
pip install -e ".[explain]"
python explain.py --bundle mtmm_release.pt --input my_compounds.csv --species human --mode both
```

Produces descriptor-level SHAP plots (Supplementary Method S5) and per-bond EdgeSHAPer
attributions drawn on the molecule (Supplementary Method S6).

### Train

`main.ipynb` runs scaffold-disjoint 10-fold cross-validation from a single CSV and writes
fold-wise metrics, a mean±SD summary, and out-of-fold predictions. Set the data path in
the final cell.

## Model and data

The trained model is distributed as a release bundle `mtmm_release.pt` (model + descriptor
scaler, imputation values, feature schema, and per-species thresholds), built with
`export_release_bundle.py`. The training and external-validation datasets are provided in
this repository.

## Contact

- munsu931122@jnu.ac.kr
- syyoo@jnu.ac.kr
