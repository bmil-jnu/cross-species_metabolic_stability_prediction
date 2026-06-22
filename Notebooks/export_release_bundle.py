#!/usr/bin/env python3
# export_release_bundle.py
# ============================================================
# Tool to build a single release bundle (mtmm_release.pt).
#
# predict.py needs the following four items together for inference:
#   (1) trained model (architecture + weights)
#   (2) descriptor column schema      desc_cols
#   (3) descriptor StandardScaler   desc_scaler        (fitted on the training fold)
#   (4) descriptor imputation values  desc_impute_values
#   (+) per-species decision thresholds  thresholds
# This script packs the above into a single .pt file.
#
# ------------------------------------------------------------
# Which model to ship (choose one)
#   (A) one representative CV fold:
#       - that fold's model (state_dict loaded) and
#         that fold's fold_info["desc_cols"/"desc_scaler"/"desc_impute_values"],
#         and that fold's thresholds.
#       - note: the model and desc_cols/scaler must come from the same fold
#         (descriptor dim must equal the model's desc_in_dim).
#   (B) one model retrained on all data (recommended for release):
#       - retrain on the full train set and pass the scaler/cols/impute fitted then.
#       - use thresholds chosen on a separate validation set.
#
# This choice is up to the author; the script does not decide it.
# ------------------------------------------------------------
#
# Example use inside the training notebook:
#   from export_release_bundle import export_bundle
#   export_bundle(
#       model=base_model,                              # MTMM with state_dict loaded
#       desc_cols=fold_info["desc_cols"],
#       desc_scaler=fold_info["desc_scaler"],
#       desc_impute_values=fold_info["desc_impute_values"],
#       thresholds=fold_thresholds,                    # e.g., {"human":.., "rat":.., "mouse":..}
#       out_path="mtmm_release.pt",
#   )
# ============================================================

import torch


def export_bundle(
    model,                  # trained MTMM (state_dict loaded; eval recommended)
    desc_cols,              # descriptor column list fitted at training time
    desc_scaler,            # sklearn StandardScaler fitted at training time
    desc_impute_values,     # imputation-value dict fitted at training time
    thresholds,             # per-species threshold dict
    out_path="mtmm_release.pt",
    tasks=("human", "rat", "mouse"),
    fp_type="morgan+maccs+rdit",
):
    # save on CPU so it loads on either cpu or gpu
    model = model.to("cpu").eval()

    # check model/desc dim match (catch mismatch early; it would error at inference)
    desc_in_dim = int(getattr(model, "desc_in_dim", -1))
    if desc_in_dim >= 0 and desc_in_dim != len(desc_cols):
        raise ValueError(
            f"[export] desc dim mismatch: model.desc_in_dim={desc_in_dim} "
            f"!= len(desc_cols)={len(desc_cols)}. "
            f"Make sure the model and desc_cols/scaler come from the same fold (or training run)."
        )

    bundle = {
        "model": model,                          # save the full module -> no need to rebuild constructor args
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
