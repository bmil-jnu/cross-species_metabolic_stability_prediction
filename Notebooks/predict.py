#!/usr/bin/env python3
# predict.py
# ============================================================
# MTMM microsomal stability prediction (inference CLI)
#   Input  : CSV with SMILES + SwissADME descriptor columns
#   Output : per-species (human/rat/mouse) stability probability + binary prediction CSV
#
# Design
#   - Reuse the training preprocessing functions
#     (model.py / dataset_scaffold_modelready.py), so inference uses the
#     same graph / fingerprint / descriptor pipeline as training (no distribution shift).
#   - Descriptors must use the schema (desc_cols) / StandardScaler /
#     impute_values fitted at training time (packed by export_release_bundle.py).
#
# Example:
#   python predict.py --bundle mtmm_release.pt \
#                     --input my_compounds.csv \
#                     --output predictions.csv
# ============================================================

import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader

# Training-pipeline modules (must sit in the same directory as predict.py)
import model  # noqa: F401  (needed so torch.load can resolve model.MTMM and related classes)
from dataset_scaffold_modelready import (
    build_desc_df_scaled,
    dataframe_to_data_list,
    find_smiles_column,
)


def load_bundle(path, device):
    """Load the release bundle (.pt) produced by export_release_bundle.py."""
    try:
        bundle = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        bundle = torch.load(path, map_location=device)

    required = [
        "model", "tasks", "desc_cols",
        "desc_scaler", "desc_impute_values", "thresholds",
    ]
    missing = [k for k in required if k not in bundle]
    if missing:
        raise KeyError(f"[predict] Missing keys in bundle: {missing}")
    return bundle


def _sync_device(model_obj, device):
    """Sync internal device attributes (saved device may differ from runtime device)."""
    model_obj = model_obj.to(device)
    if hasattr(model_obj, "device"):
        model_obj.device = device
    for m in model_obj.modules():
        if hasattr(m, "device"):
            m.device = device
    return model_obj.eval()


@torch.no_grad()
def run_inference(bundle, df, device, batch_size=64):
    tasks = list(bundle["tasks"])                 # e.g., ["human", "rat", "mouse"]
    model_obj = _sync_device(bundle["model"], device)

    # reset so the iterrows label index matches the desc_df.iloc positional index
    df = df.reset_index(drop=True).copy()

    # --- find the SMILES column (Cano_Smile / SMILES / PUBCHEM_EXT_DATASOURCE_SMILES) ---
    smi_col = find_smiles_column(df)
    if smi_col is None:
        raise ValueError(
            "[predict] SMILES column not found. "
            "Allowed headers: Cano_Smile / SMILES / PUBCHEM_EXT_DATASOURCE_SMILES"
        )

    # --- inference has no labels: inject dummy task columns (for data.y; unused in prediction) ---
    for t in tasks:
        if t not in df.columns:
            df[t] = 0.0

    # --- descriptor preprocessing: must use the training-fitted schema / scaler / impute ---
    desc_df, _, _, _ = build_desc_df_scaled(
        df,
        tasks=tasks,
        fixed_cols=bundle["desc_cols"],
        scaler=bundle["desc_scaler"],
        impute_values=bundle["desc_impute_values"],
        add_missing_indicators=True,
    )

    # --- SMILES -> list of PyG Data with graph / fingerprint / desc ---
    data_list, _ = dataframe_to_data_list(
        df=df, tasks=tasks, desc_df=desc_df, smi_col=smi_col, logger=None
    )
    if len(data_list) == 0:
        raise ValueError("[predict] No valid molecules (all SMILES failed to parse).")

    loader = GeometricDataLoader(data_list, batch_size=batch_size, shuffle=False)

    all_probs, all_smiles = [], []
    for batch in loader:
        batch = batch.to(device)
        fp = batch.smil2vec.long() if hasattr(batch, "smil2vec") else None
        _, task_outputs = model_obj({"fp": fp, "graph": batch, "desc": batch.desc})
        # task_outputs: tuple of (B,1) logits, ordered as tasks
        probs = torch.cat([torch.sigmoid(o).view(-1, 1) for o in task_outputs], dim=1)
        all_probs.append(probs.cpu().numpy())
        all_smiles.extend(list(batch.smiles))

    probs = np.concatenate(all_probs, axis=0)     # (N, num_tasks)

    # --- result table ---
    out = pd.DataFrame({"smiles": all_smiles})
    thr = bundle["thresholds"]
    for j, t in enumerate(tasks):
        t_thr = float(thr.get(t, 0.5)) if isinstance(thr, dict) else 0.5
        out[f"prob_{t}"] = probs[:, j]
        # label convention: 0 = stable, 1 = unstable  ->  pred == 1 means unstable
        out[f"pred_{t}"] = (probs[:, j] >= t_thr).astype(int)
    return out


def main():
    ap = argparse.ArgumentParser(description="MTMM microsomal stability prediction")
    ap.add_argument("--bundle", required=True, help="path to the release bundle .pt")
    ap.add_argument("--input", required=True, help="input CSV (SMILES + SwissADME columns)")
    ap.add_argument("--output", default="predictions.csv", help="output CSV path")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto-detect)")
    args = ap.parse_args()

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    bundle = load_bundle(args.bundle, device)
    df = pd.read_csv(args.input)
    out = run_inference(bundle, df, device, batch_size=args.batch_size)
    out.to_csv(args.output, index=False)
    print(f"[predict] {len(out)} molecules -> {args.output}")
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
