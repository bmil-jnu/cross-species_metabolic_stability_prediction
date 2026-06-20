#!/usr/bin/env python3
# predict.py
# ============================================================
# MTMM microsomal stability prediction (inference CLI)
#   입력 : SMILES + SwissADME descriptor 컬럼이 포함된 CSV
#   출력 : 종(human/rat/mouse)별 안정성 확률 + 이진 예측 CSV
#
# 설계 원칙
#   - 학습 코드(model.py / dataset_scaffold_modelready.py)의 전처리 함수를
#     그대로 재사용한다. 따라서 학습 때와 "동일한" graph / fingerprint /
#     descriptor 파이프라인으로 추론이 수행되어 분포 불일치가 없다.
#   - descriptor 는 반드시 학습 시 fit 한 schema(desc_cols) / StandardScaler /
#     impute_values 를 사용한다. (export_release_bundle.py 가 이를 번들에 담음)
#
# 사용 예:
#   python predict.py --bundle mtmm_release.pt \
#                     --input my_compounds.csv \
#                     --output predictions.csv
# ============================================================

import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader

# 학습 파이프라인 모듈 (predict.py 와 같은 디렉터리에 있어야 함)
import model  # noqa: F401  (torch.load 가 model.MTMM 등 클래스를 해석하는 데 필요)
from dataset_scaffold_modelready import (
    build_desc_df_scaled,
    dataframe_to_data_list,
    find_smiles_column,
)


def load_bundle(path, device):
    """export_release_bundle.py 로 만든 배포 번들(.pt)을 로드."""
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
        raise KeyError(f"[predict] 번들에 누락된 키: {missing}")
    return bundle


def _sync_device(model_obj, device):
    """저장 시점 디바이스와 런타임 디바이스가 다를 수 있으므로 내부 device 속성을 동기화."""
    model_obj = model_obj.to(device)
    if hasattr(model_obj, "device"):
        model_obj.device = device
    for m in model_obj.modules():
        if hasattr(m, "device"):
            m.device = device
    return model_obj.eval()


@torch.no_grad()
def run_inference(bundle, df, device, batch_size=64):
    tasks = list(bundle["tasks"])                 # 예: ["human", "rat", "mouse"]
    model_obj = _sync_device(bundle["model"], device)

    # iterrows 의 라벨 인덱스와 desc_df.iloc 의 위치 인덱스를 일치시키기 위해 reset
    df = df.reset_index(drop=True).copy()

    # --- SMILES 컬럼 탐색 (Cano_Smile / SMILES / PUBCHEM_EXT_DATASOURCE_SMILES) ---
    smi_col = find_smiles_column(df)
    if smi_col is None:
        raise ValueError(
            "[predict] SMILES 컬럼을 찾지 못함. "
            "허용 헤더: Cano_Smile / SMILES / PUBCHEM_EXT_DATASOURCE_SMILES"
        )

    # --- 추론에는 라벨이 없으므로 더미 task 컬럼 주입 (data.y 생성용; 예측에는 사용 안 함) ---
    for t in tasks:
        if t not in df.columns:
            df[t] = 0.0

    # --- descriptor 전처리: 학습 시 fit 한 schema / scaler / impute 사용 (필수) ---
    desc_df, _, _, _ = build_desc_df_scaled(
        df,
        tasks=tasks,
        fixed_cols=bundle["desc_cols"],
        scaler=bundle["desc_scaler"],
        impute_values=bundle["desc_impute_values"],
        add_missing_indicators=True,
    )

    # --- SMILES -> graph / fingerprint / desc 가 담긴 PyG Data 리스트 ---
    data_list, _ = dataframe_to_data_list(
        df=df, tasks=tasks, desc_df=desc_df, smi_col=smi_col, logger=None
    )
    if len(data_list) == 0:
        raise ValueError("[predict] 유효한 분자가 없음 (모든 SMILES 파싱 실패).")

    loader = GeometricDataLoader(data_list, batch_size=batch_size, shuffle=False)

    all_probs, all_smiles = [], []
    for batch in loader:
        batch = batch.to(device)
        fp = batch.smil2vec.long() if hasattr(batch, "smil2vec") else None
        _, task_outputs = model_obj({"fp": fp, "graph": batch, "desc": batch.desc})
        # task_outputs: (B,1) logit 튜플, 순서 = tasks
        probs = torch.cat([torch.sigmoid(o).view(-1, 1) for o in task_outputs], dim=1)
        all_probs.append(probs.cpu().numpy())
        all_smiles.extend(list(batch.smiles))

    probs = np.concatenate(all_probs, axis=0)     # (N, num_tasks)

    # --- 결과 테이블 ---
    out = pd.DataFrame({"smiles": all_smiles})
    thr = bundle["thresholds"]
    for j, t in enumerate(tasks):
        t_thr = float(thr.get(t, 0.5)) if isinstance(thr, dict) else 0.5
        out[f"prob_{t}"] = probs[:, j]
        # 학습 라벨 규약: 0 = stable, 1 = unstable  ->  pred == 1 이면 unstable
        out[f"pred_{t}"] = (probs[:, j] >= t_thr).astype(int)
    return out


def main():
    ap = argparse.ArgumentParser(description="MTMM microsomal stability prediction")
    ap.add_argument("--bundle", required=True, help="배포 번들 .pt 경로")
    ap.add_argument("--input", required=True, help="입력 CSV (SMILES + SwissADME 컬럼)")
    ap.add_argument("--output", default="predictions.csv", help="출력 CSV 경로")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="cpu / cuda (기본: 자동 감지)")
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
