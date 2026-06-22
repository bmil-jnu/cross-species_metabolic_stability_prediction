#!/usr/bin/env python3
# explain.py
# ============================================================
# Model interpretability reproduction script.
#   - descriptor_shap : Supplementary Method S5 (KernelExplainer-based ADME/physchem descriptor attribution)
#   - edgeshaper      : Supplementary Method S6 (bond-level Shapley + fragment aggregation)
#
# Both functions take the same release bundle (mtmm_release.pt) as predict.py.
#
# Extra dependencies: shap, matplotlib  (graph/fingerprint/descriptor stack same as predict.py)
#   pip install shap matplotlib
#
# Note: this file follows the documented Method S5/S6 spec from the manuscript.
#       Reconcile with the actual figures via a one-time smoke test in your
#       training environment (especially shap / rdkit drawing API versions).
#
# CLI example:
#   python explain.py --bundle mtmm_release.pt --input mols.csv \
#                     --species human --mode both --out-dir explain_out
# ============================================================

import os
import argparse
import numpy as np   # top-level imports: numpy/stdlib only (heavy libs imported lazily in functions)


# ============================================================
# 0. Generic Shapley estimator (permutation sampling)
#    - value_fn_batch(coalitions) -> np.ndarray  (coalitions: list[tuple[int,...]])
#    - works for any value function and exactly satisfies the efficiency axiom
#      sum_j phi_j == v(full) - v(empty) (regardless of sample count).
# ============================================================

def shapley_permutation(n_units, value_fn_batch, num_samples=200, seed=0):
    """Permutation-sampling Shapley value estimation.

    Each permutation evaluates its prefixes (partial coalitions) at once,
    so one permutation needs a single value_fn_batch call (n_units+1 coalitions).
    """
    rng = np.random.default_rng(seed)
    phi = np.zeros(n_units, dtype=np.float64)

    if n_units == 0:
        return phi

    for _ in range(num_samples):
        perm = rng.permutation(n_units)

        coalitions = [tuple()]            # empty set
        cur = []
        for u in perm:
            cur.append(int(u))
            coalitions.append(tuple(sorted(cur)))

        vals = np.asarray(value_fn_batch(coalitions), dtype=np.float64)  # (n_units+1,)
        # marginal contribution of perm[k] = vals[k+1] - vals[k]
        marg = vals[1:] - vals[:-1]
        phi[perm] += marg

    phi /= float(num_samples)
    return phi


# ============================================================
# 1. Load bundle / sync device (same behavior as predict.py)
# ============================================================

def _load(bundle_path, device):
    import torch
    import model  # noqa: F401  (needed so the bundle unpickles model.MTMM)
    try:
        bundle = torch.load(bundle_path, map_location=device, weights_only=False)
    except TypeError:
        bundle = torch.load(bundle_path, map_location=device)

    m = bundle["model"].to(device)
    if hasattr(m, "device"):
        m.device = device
    for sub in m.modules():
        if hasattr(sub, "device"):
            sub.device = device
    bundle["model"] = m.eval()
    return bundle


def _task_index(tasks, species):
    species = str(species).lower()
    tasks_l = [str(t).lower() for t in tasks]
    if species not in tasks_l:
        raise ValueError(f"[explain] species='{species}' not in tasks {tasks}.")
    return tasks_l.index(species)


def _build_data_list(bundle, df):
    """Input df -> list of PyG Data (same preprocessing as predict.py)."""
    from dataset_scaffold_modelready import build_desc_df_scaled, dataframe_to_data_list, find_smiles_column

    tasks = list(bundle["tasks"])
    df = df.reset_index(drop=True).copy()
    smi_col = find_smiles_column(df)
    if smi_col is None:
        raise ValueError("[explain] SMILES column not found (Cano_Smile/SMILES/PUBCHEM_EXT_DATASOURCE_SMILES)")
    for t in tasks:
        if t not in df.columns:
            df[t] = 0.0

    desc_df, _, _, _ = build_desc_df_scaled(
        df, tasks=tasks,
        fixed_cols=bundle["desc_cols"],
        scaler=bundle["desc_scaler"],
        impute_values=bundle["desc_impute_values"],
        add_missing_indicators=True,
    )
    data_list, _ = dataframe_to_data_list(df=df, tasks=tasks, desc_df=desc_df, smi_col=smi_col, logger=None)
    if len(data_list) == 0:
        raise ValueError("[explain] No valid molecules")
    return data_list


def _full_logits(model, data_list, device, task_idx, batch_size=256):
    """Full-model logit for the selected task -> (N,) numpy. Used for quantile sampling."""
    import torch
    from torch_geometric.loader import DataLoader as GeometricDataLoader
    loader = GeometricDataLoader(data_list, batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            fp = batch.smil2vec.long() if hasattr(batch, "smil2vec") else None
            _, task_outputs = model({"fp": fp, "graph": batch, "desc": batch.desc})
            out.append(task_outputs[task_idx].view(-1).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def select_by_logit_quantiles(logits, n):
    """Pick n indices evenly spaced over logit quantiles (Method S5)."""
    n = int(min(n, len(logits)))
    order = np.argsort(logits)                       # ascending
    pos = np.linspace(0, len(order) - 1, num=n).round().astype(int)
    pos = np.unique(pos)
    return order[pos]


# ============================================================
# 2. Method S5 — descriptor-level KernelSHAP
#    graph/fp fixed at a reference context; only desc varies.
# ============================================================

def descriptor_shap(
    bundle, df, species,
    n_background=64, n_eval=256, nsamples=2048, chunk=512,
    reference="median", device=None, seed=0,
):
    """Returns: (shap_values (n_eval, D), eval_X (n_eval, D), feature_names list[str], expected_value float)"""
    import torch
    from torch_geometric.data import Batch

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bundle["model"]
    tasks = list(bundle["tasks"])
    feature_names = list(bundle["desc_cols"])
    task_idx = _task_index(tasks, species)

    data_list = _build_data_list(bundle, df)
    desc_mat = torch.cat([d.desc for d in data_list], dim=0).float().cpu().numpy()  # (N, D) scaled

    # --- quantile-based background / eval selection ---
    logits = _full_logits(model, data_list, device, task_idx)
    bg_idx = select_by_logit_quantiles(logits, n_background)
    ev_idx = select_by_logit_quantiles(logits, n_eval)
    background = desc_mat[bg_idx]
    eval_X = desc_mat[ev_idx]

    # --- choose reference molecule (graph+fp): default is the median-logit molecule ---
    if reference == "median":
        ref_pos = int(np.argsort(logits)[len(logits) // 2])
    else:
        ref_pos = int(reference)
    ref_data = data_list[ref_pos]

    expected_desc = int(getattr(model, "desc_in_dim", desc_mat.shape[1]))

    # --- prediction function for KernelExplainer: desc matrix -> selected-task probability ---
    def f(desc_block):
        desc_block = np.asarray(desc_block, dtype=np.float32)
        if desc_block.shape[1] != expected_desc:
            # pad/trim to the model dim (same safeguard as predict)
            if desc_block.shape[1] < expected_desc:
                pad = np.zeros((desc_block.shape[0], expected_desc - desc_block.shape[1]), np.float32)
                desc_block = np.concatenate([desc_block, pad], axis=1)
            else:
                desc_block = desc_block[:, :expected_desc]

        probs = np.empty(desc_block.shape[0], dtype=np.float32)
        with torch.no_grad():
            for s in range(0, desc_block.shape[0], chunk):
                blk = desc_block[s:s + chunk]
                K = blk.shape[0]
                batch = Batch.from_data_list([ref_data] * K).to(device)  # fixed graph/fp reference
                desc_t = torch.from_numpy(blk).to(device).float()
                fp = batch.smil2vec.long() if hasattr(batch, "smil2vec") else None
                _, task_outputs = model({"fp": fp, "graph": batch, "desc": desc_t})
                probs[s:s + K] = torch.sigmoid(task_outputs[task_idx].view(-1)).detach().cpu().numpy()
        return probs

    import shap
    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(eval_X, nsamples=nsamples, l1_reg="num_features(%d)" % eval_X.shape[1])
    shap_values = np.asarray(shap_values)
    return shap_values, eval_X, feature_names, float(np.asarray(explainer.expected_value).reshape(-1)[0])


def plot_descriptor_shap(shap_values, eval_X, feature_names, species, out_dir, top_k=20):
    """beeswarm + mean|SHAP| bar (Method S5, Fig. 5 style)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    os.makedirs(out_dir, exist_ok=True)

    # beeswarm
    plt.figure()
    shap.summary_plot(shap_values, eval_X, feature_names=feature_names, max_display=top_k, show=False)
    plt.title(f"Descriptor SHAP (beeswarm) — {species.upper()}")
    bee = os.path.join(out_dir, f"shap_beeswarm_{species}.png")
    plt.tight_layout(); plt.savefig(bee, dpi=200, bbox_inches="tight"); plt.close()

    # mean|SHAP| bar
    imp = np.abs(shap_values).mean(axis=0)
    order = np.argsort(imp)[::-1][:top_k]
    plt.figure(figsize=(6, max(3, 0.32 * len(order))))
    plt.barh([feature_names[i] for i in order][::-1], imp[order][::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top descriptors — {species.upper()}")
    bar = os.path.join(out_dir, f"shap_bar_{species}.png")
    plt.tight_layout(); plt.savefig(bar, dpi=200, bbox_inches="tight"); plt.close()
    return bee, bar


# ============================================================
# 3. Method S6 — EdgeSHAPer (bond-level Shapley)
#    estimate per-bond contribution to the selected-task probability via edge perturbation.
#    bond i  <->  edge_index columns [2i, 2i+1] (construction order in dataframe_to_data_list)
# ============================================================

def edgeshaper(bundle, df_row, species, num_samples=200, device=None, seed=0):
    """Single-molecule EdgeSHAPer.

    Returns: dict {
        'mol': RDKit Mol,
        'bond_shap_unstable': np.ndarray (n_bonds,)  # raw Shapley w.r.t. P(unstable)
        'bond_shap_stabilizing': np.ndarray          # sign-flipped for viz (positive = stabilizing)
        'base_value': float, 'full_value': float, 'species': str
    }
    """
    import torch
    import pandas as pd
    from rdkit import Chem
    from torch_geometric.data import Batch

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bundle["model"]
    tasks = list(bundle["tasks"])
    task_idx = _task_index(tasks, species)

    if isinstance(df_row, pd.Series):
        df_row = df_row.to_frame().T
    data_list = _build_data_list(bundle, df_row)
    data = data_list[0]                      # single molecule
    mol = Chem.MolFromSmiles(data.smiles)
    n_bonds = mol.GetNumBonds()

    # directed edge rows of bond i: [2i, 2i+1]
    def _masked_data(bond_subset):
        d = data.clone()
        if len(bond_subset) == 0:
            d.edge_index = torch.empty((2, 0), dtype=torch.long)
            d.edge_attr = torch.empty((0, data.edge_attr.size(1)), dtype=torch.float)
        else:
            rows = []
            for b in bond_subset:
                rows.extend([2 * b, 2 * b + 1])
            rows = torch.tensor(sorted(rows), dtype=torch.long)
            d.edge_index = data.edge_index[:, rows]
            d.edge_attr = data.edge_attr[rows]
        return d

    @torch.no_grad()
    def value_fn_batch(coalitions):
        ds = [_masked_data(c) for c in coalitions]
        batch = Batch.from_data_list(ds).to(device)
        fp = batch.smil2vec.long() if hasattr(batch, "smil2vec") else None
        _, task_outputs = model({"fp": fp, "graph": batch, "desc": batch.desc})
        return torch.sigmoid(task_outputs[task_idx].view(-1)).detach().cpu().numpy()

    phi = shapley_permutation(n_bonds, value_fn_batch, num_samples=num_samples, seed=seed)

    base_value = float(value_fn_batch([tuple()])[0])                 # all edges removed
    full_value = float(value_fn_batch([tuple(range(n_bonds))])[0])   # all edges present

    # model positive class = unstable (label 1). Negate for the viz convention (positive = stabilizing).
    return {
        "mol": mol,
        "bond_shap_unstable": phi,
        "bond_shap_stabilizing": -phi,
        "base_value": base_value,
        "full_value": full_value,
        "species": species,
    }


def draw_edge_attribution(result, out_path, size=(520, 420)):
    """Color per-bond contributions (stabilizing sign) onto the molecule.
       blue = stabilizing (positive), red = destabilizing (negative). (Fig. 6 convention)"""
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = result["mol"]
    scores = np.asarray(result["bond_shap_stabilizing"], dtype=float)
    n_bonds = mol.GetNumBonds()
    if n_bonds == 0:
        return None

    vmax = np.max(np.abs(scores)) or 1.0
    highlight_bonds = list(range(n_bonds))
    bond_colors = {}
    for b in range(n_bonds):
        t = float(scores[b]) / vmax            # [-1, 1]
        if t >= 0:                              # stabilizing -> blue
            bond_colors[b] = (1.0 - t, 1.0 - t, 1.0)
        else:                                   # destabilizing -> red
            t = -t
            bond_colors[b] = (1.0, 1.0 - t, 1.0 - t)

    d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=[], highlightBonds=highlight_bonds, highlightBondColors=bond_colors,
    )
    d.FinishDrawing()
    with open(out_path, "wb") as fh:
        fh.write(d.GetDrawingText())
    return out_path


# ============================================================
# 4. CLI
# ============================================================

def main():
    import torch
    import pandas as pd

    ap = argparse.ArgumentParser(description="MTMM model explanation (SHAP / EdgeSHAPer)")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--input", required=True, help="CSV (SMILES + SwissADME columns)")
    ap.add_argument("--species", default="human", help="human / rat / mouse")
    ap.add_argument("--mode", default="both", choices=["descriptor", "edge", "both"])
    ap.add_argument("--out-dir", default="explain_out")
    ap.add_argument("--n-background", type=int, default=64)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--nsamples", type=int, default=2048)
    ap.add_argument("--edge-samples", type=int, default=200)
    ap.add_argument("--edge-rows", type=int, default=3, help="number of top input rows to draw edge explanations for")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    bundle = _load(args.bundle, device)
    df = pd.read_csv(args.input)

    if args.mode in ("descriptor", "both"):
        sv, ex, names, ev = descriptor_shap(
            bundle, df, args.species,
            n_background=args.n_background, n_eval=args.n_eval,
            nsamples=args.nsamples, device=device,
        )
        bee, bar = plot_descriptor_shap(sv, ex, names, args.species, args.out_dir)
        print(f"[descriptor SHAP] -> {bee} , {bar}  (E[f]={ev:.4f})")

    if args.mode in ("edge", "both"):
        for i in range(min(args.edge_rows, len(df))):
            res = edgeshaper(bundle, df.iloc[i], args.species, num_samples=args.edge_samples, device=device)
            png = os.path.join(args.out_dir, f"edgeshaper_{args.species}_row{i}.png")
            draw_edge_attribution(res, png)
            print(f"[EdgeSHAPer] row{i} base={res['base_value']:.3f} full={res['full_value']:.3f} -> {png}")


if __name__ == "__main__":
    main()
