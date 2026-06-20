#!/usr/bin/env python3
# explain.py
# ============================================================
# 모델 설명(interpretability) 재현 스크립트.
#   - descriptor_shap : Supplementary Method S5 (KernelExplainer 기반 ADME/물리화학 descriptor 기여도)
#   - edgeshaper      : Supplementary Method S6 (bond 단위 Shapley + fragment 집계)
#
# 두 함수 모두 predict.py 와 동일한 배포 번들(mtmm_release.pt)을 입력으로 받는다.
#
# 추가 의존성: shap, matplotlib  (그래프/지문/descriptor 스택은 predict.py 와 동일)
#   pip install shap matplotlib
#
# 주의: 본 파일은 매뉴스크립트 Method S5/S6 의 "문서화된 사양"에 맞춰 작성한
#       재현 구현이다. 실제 그림과의 정합은 사용자의 학습 환경에서 1회 스모크
#       테스트로 확인할 것(특히 shap/rdkit drawing API 버전).
#
# CLI 예:
#   python explain.py --bundle mtmm_release.pt --input mols.csv \
#                     --species human --mode both --out-dir explain_out
# ============================================================

import os
import argparse
import numpy as np   # 상위 임포트는 numpy/stdlib 만 (heavy 라이브러리는 함수 내부에서 lazy 임포트)


# ============================================================
# 0. 공용 Shapley 추정기 (permutation sampling)
#    - value_fn_batch(coalitions) -> np.ndarray  (coalitions: list[tuple[int,...]])
#    - 임의의 value function 에 대해 동작하며, 효율성 공리
#      sum_j phi_j == v(전체) - v(공집합) 를 (표본 수와 무관하게) 정확히 만족한다.
# ============================================================

def shapley_permutation(n_units, value_fn_batch, num_samples=200, seed=0):
    """순열 표본 기반 Shapley 값 추정.

    각 순열에서 prefix(부분 연합)들을 한 번에 평가하므로,
    한 순열당 value_fn_batch 호출 1회(연합 n_units+1 개)로 끝난다.
    """
    rng = np.random.default_rng(seed)
    phi = np.zeros(n_units, dtype=np.float64)

    if n_units == 0:
        return phi

    for _ in range(num_samples):
        perm = rng.permutation(n_units)

        coalitions = [tuple()]            # 공집합
        cur = []
        for u in perm:
            cur.append(int(u))
            coalitions.append(tuple(sorted(cur)))

        vals = np.asarray(value_fn_batch(coalitions), dtype=np.float64)  # (n_units+1,)
        # perm[k] 의 한계 기여 = vals[k+1] - vals[k]
        marg = vals[1:] - vals[:-1]
        phi[perm] += marg

    phi /= float(num_samples)
    return phi


# ============================================================
# 1. 번들 로드 / 디바이스 동기화 (predict.py 와 동일 동작)
# ============================================================

def _load(bundle_path, device):
    import torch
    import model  # noqa: F401  (번들 unpickle 시 model.MTMM 해석에 필요)
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
        raise ValueError(f"[explain] species='{species}' 가 tasks {tasks} 에 없음")
    return tasks_l.index(species)


def _build_data_list(bundle, df):
    """입력 df -> PyG Data 리스트 (predict.py 와 동일 전처리)."""
    from dataset_scaffold_modelready import build_desc_df_scaled, dataframe_to_data_list, find_smiles_column

    tasks = list(bundle["tasks"])
    df = df.reset_index(drop=True).copy()
    smi_col = find_smiles_column(df)
    if smi_col is None:
        raise ValueError("[explain] SMILES 컬럼을 찾지 못함 (Cano_Smile/SMILES/PUBCHEM_EXT_DATASOURCE_SMILES)")
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
        raise ValueError("[explain] 유효한 분자가 없음")
    return data_list


def _full_logits(model, data_list, device, task_idx, batch_size=256):
    """전체 모델 logit (선택 task) -> (N,) numpy. 분위수 표본 선택용."""
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
    """logit 분위수로 n 개 인덱스를 균등 간격으로 선택 (Method S5)."""
    n = int(min(n, len(logits)))
    order = np.argsort(logits)                       # 오름차순
    pos = np.linspace(0, len(order) - 1, num=n).round().astype(int)
    pos = np.unique(pos)
    return order[pos]


# ============================================================
# 2. Method S5 — descriptor-level KernelSHAP
#    graph/fp 는 reference context 로 고정, desc 만 변화.
# ============================================================

def descriptor_shap(
    bundle, df, species,
    n_background=64, n_eval=256, nsamples=2048, chunk=512,
    reference="median", device=None, seed=0,
):
    """반환: (shap_values (n_eval, D), eval_X (n_eval, D), feature_names list[str], expected_value float)"""
    import torch
    from torch_geometric.data import Batch

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bundle["model"]
    tasks = list(bundle["tasks"])
    feature_names = list(bundle["desc_cols"])
    task_idx = _task_index(tasks, species)

    data_list = _build_data_list(bundle, df)
    desc_mat = torch.cat([d.desc for d in data_list], dim=0).float().cpu().numpy()  # (N, D) scaled

    # --- 분위수 기반 background / eval 선택 ---
    logits = _full_logits(model, data_list, device, task_idx)
    bg_idx = select_by_logit_quantiles(logits, n_background)
    ev_idx = select_by_logit_quantiles(logits, n_eval)
    background = desc_mat[bg_idx]
    eval_X = desc_mat[ev_idx]

    # --- reference 분자(graph+fp) 결정: 기본은 logit 중앙값 분자 ---
    if reference == "median":
        ref_pos = int(np.argsort(logits)[len(logits) // 2])
    else:
        ref_pos = int(reference)
    ref_data = data_list[ref_pos]

    expected_desc = int(getattr(model, "desc_in_dim", desc_mat.shape[1]))

    # --- KernelExplainer 용 예측 함수: desc 행렬 -> 선택 task 확률 ---
    def f(desc_block):
        desc_block = np.asarray(desc_block, dtype=np.float32)
        if desc_block.shape[1] != expected_desc:
            # 모델 차원에 맞춰 보정 (predict 와 동일 안전장치)
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
                batch = Batch.from_data_list([ref_data] * K).to(device)  # graph/fp 고정 reference
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
    """beeswarm + mean|SHAP| bar (Method S5, Fig.5 스타일)."""
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
# 3. Method S6 — EdgeSHAPer (bond 단위 Shapley)
#    edge perturbation 으로 선택 task 확률에 대한 bond 기여도 추정.
#    bond i  <->  edge_index 의 컬럼 [2i, 2i+1] (dataframe_to_data_list 의 생성 순서)
# ============================================================

def edgeshaper(bundle, df_row, species, num_samples=200, device=None, seed=0):
    """단일 분자 EdgeSHAPer.

    반환: dict {
        'mol': RDKit Mol,
        'bond_shap_unstable': np.ndarray (n_bonds,)  # P(unstable) 에 대한 raw Shapley
        'bond_shap_stabilizing': np.ndarray          # 시각화용 부호반전(양수=stabilizing)
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
    data = data_list[0]                      # 단일 분자
    mol = Chem.MolFromSmiles(data.smiles)
    n_bonds = mol.GetNumBonds()

    # bond i 의 directed edge 행: [2i, 2i+1]
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

    base_value = float(value_fn_batch([tuple()])[0])                 # 모든 edge 제거
    full_value = float(value_fn_batch([tuple(range(n_bonds))])[0])   # 모든 edge 포함

    # 모델 양성 클래스 = unstable(label 1). 시각화 부호규약(양수=stabilizing) 적용 위해 반전.
    return {
        "mol": mol,
        "bond_shap_unstable": phi,
        "bond_shap_stabilizing": -phi,
        "base_value": base_value,
        "full_value": full_value,
        "species": species,
    }


def draw_edge_attribution(result, out_path, size=(520, 420)):
    """bond 기여도(stabilizing 부호)를 분자 위에 색으로 표시.
       blue = stabilizing(양수), red = destabilizing(음수). (Fig.6 규약)"""
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
    ap.add_argument("--input", required=True, help="CSV (SMILES + SwissADME 컬럼)")
    ap.add_argument("--species", default="human", help="human / rat / mouse")
    ap.add_argument("--mode", default="both", choices=["descriptor", "edge", "both"])
    ap.add_argument("--out-dir", default="explain_out")
    ap.add_argument("--n-background", type=int, default=64)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--nsamples", type=int, default=2048)
    ap.add_argument("--edge-samples", type=int, default=200)
    ap.add_argument("--edge-rows", type=int, default=3, help="edge 설명을 그릴 입력 상위 행 개수")
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
