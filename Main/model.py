#!/usr/bin/env python3
# export_release_bundle.py
# ============================================================
# 배포용 단일 번들(mtmm_release.pt) 생성 도구.
#
# predict.py 가 추론하려면 다음 4가지가 "한 세트로" 필요하다.
#   (1) 학습된 모델 (architecture + weights)
#   (2) descriptor 컬럼 스키마      desc_cols
#   (3) descriptor StandardScaler   desc_scaler        ← 학습 fold 에서 fit 한 것
#   (4) descriptor 결측 대치값       desc_impute_values
#   (+) 종별 결정 임계값             thresholds
# 이 스크립트는 위 항목을 하나의 .pt 로 묶는다.
#
# ------------------------------------------------------------
# 어떤 모델을 배포할지 (둘 중 하나 선택)
#   (A) CV fold 중 대표 1개:
#       - 해당 fold 의 모델(state_dict 로드 완료)과
#         "그 fold 의" fold_info["desc_cols"/"desc_scaler"/"desc_impute_values"],
#         "그 fold 의" thresholds 를 함께 넘긴다.
#       - 주의: 모델과 desc_cols/scaler 는 반드시 같은 fold 의 것이어야 한다
#         (desc 차원 = 모델 desc_in_dim 이 일치해야 함).
#   (B) 전체 데이터 재학습 1개 (배포 권장):
#       - 전체 train 으로 한 번 더 학습하고, 그때 fit 한 scaler/cols/impute 를 넘긴다.
#       - 임계값은 별도 validation 에서 고른 값을 사용.
#
# 이 부분은 munsubang 이 직접 선택해야 하며, 스크립트가 임의로 정하지 않는다.
# ------------------------------------------------------------
#
# 학습 노트북 안에서의 사용 예:
#   from export_release_bundle import export_bundle
#   export_bundle(
#       model=base_model,                              # state_dict 로드 완료된 MTMM
#       desc_cols=fold_info["desc_cols"],
#       desc_scaler=fold_info["desc_scaler"],
#       desc_impute_values=fold_info["desc_impute_values"],
#       thresholds=fold_thresholds,                    # 예: {"human":.., "rat":.., "mouse":..}
#       out_path="mtmm_release.pt",
#   )
# ============================================================

import torch


def export_bundle(
    model,                  # 학습된 MTMM (state_dict 로드 완료, eval 권장)
    desc_cols,              # 학습 fit 한 descriptor 컬럼 리스트
    desc_scaler,            # 학습 fit 한 sklearn StandardScaler
    desc_impute_values,     # 학습 fit 한 결측 대치값 dict
    thresholds,             # 종별 임계값 dict
    out_path="mtmm_release.pt",
    tasks=("human", "rat", "mouse"),
    fp_type="morgan+maccs+rdit",
):
    # CPU 로 저장하면 cpu/gpu 어디서든 로드 가능
    model = model.to("cpu").eval()

    # 모델과 desc 차원 일치 점검 (불일치 시 추론에서 에러가 나므로 미리 잡는다)
    desc_in_dim = int(getattr(model, "desc_in_dim", -1))
    if desc_in_dim >= 0 and desc_in_dim != len(desc_cols):
        raise ValueError(
            f"[export] desc 차원 불일치: model.desc_in_dim={desc_in_dim} "
            f"!= len(desc_cols)={len(desc_cols)}. "
            f"모델과 desc_cols/scaler 가 같은 fold(또는 같은 학습)에서 나온 것인지 확인하세요."
        )

    bundle = {
        "model": model,                          # 전체 모듈 저장 → 생성자 인자 재현 불필요
        "tasks": list(tasks),
        "desc_cols": list(desc_cols),
        "desc_scaler": desc_scaler,
        "desc_impute_values": dict(desc_impute_values),
        "thresholds": dict(thresholds),
        "fp_type": fp_type,
    }

    torch.save(bundle, out_path)
    print(
        f"[export] saved -> {out_path}  "
        f"(desc_dim={len(desc_cols)}, tasks={list(tasks)}, thresholds={dict(thresholds)})"
    )
    return out_path
