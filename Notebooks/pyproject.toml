# pyproject.toml
# ============================================================
# pip install + CLI entry point (flat-module layout).
#   - Flat modules keep existing imports unchanged and keep the release
#     bundle's `model.MTMM` pickle path resolvable.
# Install (from repo root):  pip install -e .
# Predict:                   mtmm-predict --bundle mtmm_release.pt --input mols.csv --output preds.csv
# ============================================================

[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mtmm-stability"
version = "1.0.0"
description = "Cross-species liver microsomal metabolic stability predictor (multi-task multi-modal)."
readme = "README.md"
requires-python = ">=3.10"
keywords = ["metabolic-stability", "ADMET", "drug-discovery", "graph-neural-network"]
classifiers = [
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]

# Confirmed paper environment: Python 3.12.5.
# torch / torch_geometric / rdkit may need environment-specific wheels (CUDA / PyG);
# install them per your setup if `pip install` alone does not resolve.
dependencies = [
    "numpy",
    "pandas",
    "scikit-learn==1.6.1",
    "rdkit",
    "torch==2.2.0",
    "torch_geometric==2.6.1",
]

[project.optional-dependencies]
explain = ["shap", "matplotlib"]   # for explain.py (SHAP / EdgeSHAPer visualizations)

[project.urls]
Homepage = "https://github.com/bmil-jnu/cross-species_metabolic_stability_prediction"

# TODO: fill author info
# authors = [{ name = "<author>", email = "<email>" }]

[project.scripts]
# Inference CLI. export_release_bundle is library-only (needs the trained model object).
mtmm-predict = "predict:main"

[tool.setuptools]
# Inference modules only; training-only files (train.py / utile.py / Focal_loss.py) are excluded.
py-modules = ["predict", "export_release_bundle", "explain", "model", "dataset_scaffold_modelready"]
