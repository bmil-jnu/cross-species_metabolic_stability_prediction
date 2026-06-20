import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as GeometricDataLoader
from collections import defaultdict
from math import cos, pi
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, List, Dict

from dataset_scaffold_modelready import build_scaffold_kfold_loader, MolDataset, seq_dict_smi
from model import MTMM
from utile import (
    seed_set,
    create_logger,
    EarlyStopping,
    printPerformance,
    get_metric_func,
    build_scheduler,
    evaluate_with_fixed_threshold,
)
from Focal_loss import FocalLoss

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def _prep_fp_tensor(fp: torch.Tensor, B: int, L: int = 100) -> torch.Tensor:
    if fp.dim() == 3 and fp.size(1) == 1:
        fp = fp.squeeze(1)
    if fp.dim() == 1 and fp.numel() == B * L:
        fp = fp.view(B, L)
    return fp


def _forward_three_modal(model, data, device, logger=None):
    B = data.num_graphs

    # FP
    fp = getattr(data, "smil2vec", None)
    if fp is None:
        fp = torch.zeros((B, 100), device=device, dtype=torch.long)
    else:
        fp = _prep_fp_tensor(fp, B=B, L=100).to(device)
        if fp.dtype != torch.long:
            fp = fp.long()

    # DESC
    expected_desc = int(getattr(model, 'desc_in_dim', 0) or 0)
    desc = getattr(data, "desc", None)

    if desc is None:
        desc = torch.zeros((B, expected_desc if expected_desc > 0 else 1),
                           device=device, dtype=torch.float32)
    else:
        desc = desc.to(device).float()
        if desc.dim() == 1:
            if desc.numel() == B: 
                desc = desc.unsqueeze(1)
            elif desc.numel() % B == 0: 
                desc = desc.view(B, -1)
            else: 
                desc = desc.unsqueeze(1)
        elif desc.dim() > 2:
            desc = desc.view(B, -1)

        if expected_desc > 0:
            D = desc.size(-1)
            if D < expected_desc:
                pad = torch.zeros((B, expected_desc - D), device=device, dtype=desc.dtype)
                desc = torch.cat([desc, pad], dim=1)
            elif D > expected_desc:
                desc = desc[:, :expected_desc]

    # GRAPH
    pooled, task_outputs = model({'fp': fp, 'graph': data, 'desc': desc})
    return pooled, task_outputs


def make_cosine_with_warmup(optimizer, max_epochs, warmup_epochs=5, min_lr=1e-6, base_lr=1e-4):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        t = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        cos_decay = 0.5 * (1 + cos(pi * t))
        return max(min_lr / base_lr, cos_decay)
    return LambdaLR(optimizer, lr_lambda)


def compute_pos_weight_per_task(loaders, num_tasks, device):
    pos = torch.zeros(num_tasks, dtype=torch.float64)
    neg = torch.zeros(num_tasks, dtype=torch.float64)
    for loader in loaders:
        for batch in loader:
            y = batch.y
            if y.dim() == 1: 
                y = y.view(-1, num_tasks)
            for i in range(num_tasks):
                valid = (y[:, i] != -1)
                if valid.any():
                    yi = y[valid, i]
                    pos[i] += (yi == 1).sum().item()
                    neg[i] += (yi == 0).sum().item()
    pos = torch.clamp(pos, min=1.0)
    w = (neg / pos).to(torch.float32)
    return w.to(device)


def compute_global_max_token_idx(datasets):
    gmax = 0
    for ds in datasets:
        for d in ds:
            if hasattr(d, 'smil2vec') and d.smil2vec is not None:
                flat = d.smil2vec.view(-1)
                if flat.numel() == 0: 
                    continue
                v = int(flat.max().item())
                gmax = max(gmax, v)
    return gmax


def _infer_desc_dim(train_desc_cols, tr_ds, test_ds) -> int:
    if train_desc_cols is not None:
        return len(train_desc_cols)
    for ds in (tr_ds, test_ds):
        if len(ds) > 0 and hasattr(ds[0], "desc") and getattr(ds[0], "desc") is not None:
            d = ds[0].desc
            return int(d.view(d.size(0), -1).size(1)) if d.dim() > 1 else int(d.size(-1))
    return 0


def _save_parquet_or_csv(df, path, logger=None):
    base, _ = os.path.splitext(path)
    csv_path = base + ".csv"
    df.to_csv(csv_path, index=False)
    if logger: 
        logger.info(f"[SAVE] Predictions -> {csv_path}")
    return csv_path
 

# ------------------------------------------------------------------------------
# Neural + Tanimoto kNN Blending Helpers
# ------------------------------------------------------------------------------
@torch.no_grad()
def collect_neural_scores_full(
    model,
    loader,
    device,
    num_tasks,
):
    """
    loader 순서 그대로 label matrix와 neural probability matrix를 반환합니다.

    Returns
    -------
    y_mat : np.ndarray, shape (N, num_tasks)
    score_mat : np.ndarray, shape (N, num_tasks)
        sigmoid probability for classification.
    """
    model.eval()

    y_list = []
    score_list = []

    for batch in loader:
        batch = batch.to(device)

        _, task_outputs = _forward_three_modal(
            model,
            batch,
            device,
            logger=None,
        )

        y = batch.y
        if y.dim() == 1:
            y = y.view(-1, num_tasks)

        probs = torch.cat(
            [torch.sigmoid(o).detach() for o in task_outputs],
            dim=1,
        )

        y_list.append(y.detach().cpu().numpy())
        score_list.append(probs.detach().cpu().numpy())

    if len(y_list) == 0:
        return (
            np.empty((0, num_tasks), dtype=np.float32),
            np.empty((0, num_tasks), dtype=np.float32),
        )

    y_mat = np.concatenate(y_list, axis=0)
    score_mat = np.concatenate(score_list, axis=0)

    return y_mat, score_mat


def collect_morgan_fps_and_labels(
    loader,
    num_tasks,
):
    """
    PyG loader에서 Morgan fingerprint와 label matrix를 수집합니다.

    Returns
    -------
    fps : np.ndarray, shape (N, 2048), uint8
    y_mat : np.ndarray, shape (N, num_tasks), float
    """
    fp_list = []
    y_list = []

    for batch in loader:
        if not hasattr(batch, "morgan_fp"):
            raise AttributeError(
                "batch에 morgan_fp가 없습니다. "
                "dataset에서 data.morgan_fp를 생성하는지 확인하세요."
            )

        fp = batch.morgan_fp.detach().cpu()

        if fp.dim() == 3 and fp.size(1) == 1:
            fp = fp.squeeze(1)
        elif fp.dim() == 1:
            fp = fp.view(1, -1)
        elif fp.dim() > 3:
            fp = fp.view(fp.size(0), -1)

        y = batch.y.detach().cpu()
        if y.dim() == 1:
            y = y.view(-1, num_tasks)

        fp_list.append((fp.numpy() > 0).astype(np.uint8))
        y_list.append(y.numpy())

    if len(fp_list) == 0:
        return (
            np.empty((0, 2048), dtype=np.uint8),
            np.empty((0, num_tasks), dtype=np.float32),
        )

    fps = np.concatenate(fp_list, axis=0)
    y_mat = np.concatenate(y_list, axis=0)

    return fps, y_mat


def compute_tanimoto_knn_scores_by_task(
    query_fps,
    ref_fps,
    ref_y,
    num_tasks,
    k=5,
    chunk_size=512,
    device=None,
    eps=1e-8,
):
    """
    Tanimoto similarity 기반 kNN score를 task별로 계산합니다.

    score = top-k reference labels의 similarity-weighted average

    Parameters
    ----------
    query_fps : np.ndarray, shape (Nq, D)
    ref_fps : np.ndarray, shape (Nr, D)
    ref_y : np.ndarray, shape (Nr, num_tasks)
    num_tasks : int
    k : int
    chunk_size : int
    device : torch.device or None
        None이면 cuda 사용 가능 시 cuda, 아니면 cpu.

    Returns
    -------
    knn_scores : np.ndarray, shape (Nq, num_tasks)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query_fps = (np.asarray(query_fps) > 0).astype(np.float32)
    ref_fps = (np.asarray(ref_fps) > 0).astype(np.float32)
    ref_y = np.asarray(ref_y)

    n_query = query_fps.shape[0]
    knn_scores = np.zeros((n_query, num_tasks), dtype=np.float32)

    if n_query == 0 or ref_fps.shape[0] == 0:
        knn_scores[:] = 0.5
        return knn_scores

    ref_t = torch.as_tensor(ref_fps, dtype=torch.float32, device=device)
    ref_sum = ref_t.sum(dim=1)

    ref_y_t = torch.as_tensor(ref_y, dtype=torch.float32, device=device)

    for start in range(0, n_query, chunk_size):
        end = min(start + chunk_size, n_query)

        q_t = torch.as_tensor(
            query_fps[start:end],
            dtype=torch.float32,
            device=device,
        )

        q_sum = q_t.sum(dim=1)

        inter = q_t @ ref_t.T
        union = q_sum[:, None] + ref_sum[None, :] - inter
        sim = inter / (union + eps)

        for task_idx in range(num_tasks):
            valid_ref = ref_y_t[:, task_idx] != -1

            if int(valid_ref.sum().item()) == 0:
                knn_scores[start:end, task_idx] = 0.5
                continue

            sim_t = sim[:, valid_ref]
            y_t = ref_y_t[valid_ref, task_idx]

            kk = min(int(k), sim_t.size(1))

            top_sim, top_idx = torch.topk(
                sim_t,
                k=kk,
                dim=1,
                largest=True,
                sorted=False,
            )

            top_y = y_t[top_idx]

            denom = top_sim.sum(dim=1)
            fallback = y_t.mean()

            score = (top_sim * top_y).sum(dim=1) / (denom + eps)
            score = torch.where(denom > eps, score, fallback.expand_as(score))

            knn_scores[start:end, task_idx] = (
                score.detach().cpu().numpy().astype(np.float32)
            )

        del q_t, inter, union, sim
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return knn_scores


def select_blend_weights_by_validation_auc(
    y_mat,
    neural_scores,
    knn_scores,
    tasks,
    w_grid=None,
    logger=None,
):
    """
    validation set에서 species별 best blending weight를 선택합니다.

    blended_score = (1 - w) * neural_score + w * knn_score

    Returns
    -------
    best_w_dict : dict[str, float]
    best_auc_dict : dict[str, float]
    """
    from sklearn.metrics import roc_auc_score

    if w_grid is None:
        w_grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

    y_mat = np.asarray(y_mat)
    neural_scores = np.asarray(neural_scores)
    knn_scores = np.asarray(knn_scores)

    best_w_dict = {}
    best_auc_dict = {}

    for task_idx, sp in enumerate(tasks):
        y_true = y_mat[:, task_idx]
        neural = neural_scores[:, task_idx]
        knn = knn_scores[:, task_idx]

        valid = (
            (y_true != -1)
            & np.isfinite(y_true)
            & np.isfinite(neural)
            & np.isfinite(knn)
        )

        if valid.sum() == 0 or np.unique(y_true[valid]).size < 2:
            best_w_dict[sp] = 0.0
            best_auc_dict[sp] = np.nan
            continue

        y_v = y_true[valid].astype(int)
        neural_v = neural[valid]
        knn_v = knn[valid]

        best_w = 0.0
        best_auc = -np.inf

        for w in w_grid:
            blended = (1.0 - float(w)) * neural_v + float(w) * knn_v

            try:
                auc = roc_auc_score(y_v, blended)
            except Exception:
                auc = np.nan

            if np.isfinite(auc) and auc > best_auc:
                best_auc = float(auc)
                best_w = float(w)

        best_w_dict[sp] = best_w
        best_auc_dict[sp] = best_auc

        if logger is not None:
            logger.info(
                f"[Blend Select] {sp.upper()} | "
                f"best_w={best_w:.2f} | val_auc={best_auc:.4f}"
            )

    return best_w_dict, best_auc_dict


def apply_blend_weights(
    neural_scores,
    knn_scores,
    tasks,
    blend_weights,
):
    """
    species별 blend weight를 적용해 blended score matrix를 반환합니다.

    Parameters
    ----------
    neural_scores : np.ndarray, shape (N, num_tasks)
    knn_scores : np.ndarray, shape (N, num_tasks)
    tasks : list[str]
    blend_weights : dict[str, float]

    Returns
    -------
    blended : np.ndarray, shape (N, num_tasks)
    """
    neural_scores = np.asarray(neural_scores, dtype=np.float32)
    knn_scores = np.asarray(knn_scores, dtype=np.float32)

    blended = np.zeros_like(neural_scores, dtype=np.float32)

    for i, sp in enumerate(tasks):
        w = float(blend_weights.get(sp, 0.0))
        blended[:, i] = (1.0 - w) * neural_scores[:, i] + w * knn_scores[:, i]

    return blended

# ------------------------------------------------------------------------------
# Improved MultiTask Loss Wrapper
# ------------------------------------------------------------------------------

def alpha_from_pos_weights(pos_weights, lo: float = 0.05, hi: float = 0.95):
    """
    pos_weight(w = neg / pos)를 focal alpha로 변환.
    alpha / (1 - alpha) = w
    alpha = w / (1 + w)

    예:
    pos_weight = 1.0이면 alpha = 0.5
    pos_weight > 1.0이면 positive class에 더 큰 alpha 부여
    """
    w = torch.as_tensor(pos_weights, dtype=torch.float32).reshape(-1)
    a = w / (1.0 + w)
    a = a.clamp(min=lo, max=hi)
    return [float(v) for v in a]


class ImprovedMultiTaskLossWrapper(nn.Module):
    """
    Multi-task classification loss wrapper.

    지원 기능
    ----------
    1. loss_type="bce"
       - binary_cross_entropy_with_logits 사용
       - label_smoothing 사용 가능
       - pos_weight 사용 가능

    2. loss_type="focal"
       - 기존 FocalLoss 사용
       - alpha="auto"인 경우 pos_weights 기반 alpha 자동 계산

    3. optional ranking loss
       - 같은 task 안에서 positive logit이 negative logit보다 margin만큼 크도록 유도

    4. optional uncertainty weighting
       - task별 log variance를 학습하여 task loss를 자동 가중

    5. task loss reduction
       - 기본값은 "mean"
       - valid label이 존재하는 task들의 loss를 평균냄
       - -1 mask가 많은 multitask setting에서 loss scale 안정화에 유리

    반환값
    ----------
    total_loss, pooled, task_outputs, avg_cls_loss, avg_ranking_loss
    """
    def __init__(
        self,
        model,
        num_tasks,
        loss_type: str = "bce",
        gamma: float = 2.0,
        alpha="auto",
        label_smoothing: float = 0.0,
        use_bce_pos_weight: bool = False,
        ranking_margin: float = 0.2,
        ranking_lambda: float = 0.0,
        use_ranking: bool = False,
        use_uncertainty: bool = False,
        task_loss_reduction: str = "mean",
        task_weights=None,
        forward_three_modal=None,
        focal_cls=None,
        **_ignored,
    ):
        super().__init__()

        self.model = model
        self.num_tasks = int(num_tasks)

        self.loss_type = str(loss_type).lower()
        if self.loss_type not in ("focal", "bce"):
            raise ValueError(f"loss_type must be 'focal' or 'bce', got {loss_type}")

        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.use_bce_pos_weight = bool(use_bce_pos_weight)

        self.ranking_margin = float(ranking_margin)
        self.ranking_lambda = float(ranking_lambda)
        self.use_ranking = bool(use_ranking)
        self.use_uncertainty = bool(use_uncertainty)

        self.task_loss_reduction = str(task_loss_reduction).lower()
        if self.task_loss_reduction not in ("mean", "sum"):
            raise ValueError(
                f"task_loss_reduction must be 'mean' or 'sum', "
                f"got {task_loss_reduction}"
            )
        # --------------------------------------------------
        # task-specific loss weights
        # 예: tasks = ["human", "rat", "mouse"]일 때
        # task_weights=[1.0, 1.0, 1.2]이면 mouse loss를 20% 더 반영
        # --------------------------------------------------
        if task_weights is None:
            task_weights_tensor = torch.ones(self.num_tasks, dtype=torch.float32)
        else:
            task_weights_tensor = torch.as_tensor(task_weights, dtype=torch.float32).reshape(-1)

            if task_weights_tensor.numel() != self.num_tasks:
                raise ValueError(
                    f"task_weights length ({task_weights_tensor.numel()}) "
                    f"!= num_tasks ({self.num_tasks})"
                )

        self.register_buffer("task_weights", task_weights_tensor)
        # uncertainty weighting용 task별 log variance
        # use_uncertainty=False이면 forward에서 사용하지 않음
        self.log_vars = nn.Parameter(torch.zeros(self.num_tasks))

        # 외부 forward 함수 / focal class 주입용
        self._injected_fwd = forward_three_modal
        self._focal_cls = focal_cls

        # BCE pos_weight cache
        self._bce_pos_weight = None

        # focal 관련 변수
        self.focals = None
        self._alpha_auto = False
        self._alpha_list = None

        if self.loss_type == "focal":
            self._alpha_auto = isinstance(alpha, str) and alpha.lower() == "auto"

            if self._alpha_auto:
                # 첫 forward에서 pos_weights가 들어오면 alpha 자동 생성
                self._alpha_list = None
            else:
                self._alpha_list = self._broadcast_alpha(alpha)
                self._build_focals()

    # ------------------------------------------------------------------
    def _broadcast_alpha(self, alpha):
        """
        focal alpha를 task 개수에 맞게 정리.
        alpha가 scalar이면 모든 task에 동일 적용.
        alpha가 list/tensor이면 num_tasks 길이와 일치해야 함.
        """
        if isinstance(alpha, (list, tuple)):
            seq = list(alpha)
        elif torch.is_tensor(alpha):
            seq = [float(v) for v in alpha.detach().cpu().reshape(-1)]
        else:
            seq = [float(alpha)] * self.num_tasks

        if len(seq) != self.num_tasks:
            raise ValueError(
                f"[Wrapper] alpha length ({len(seq)}) != num_tasks ({self.num_tasks})"
            )

        return [float(v) for v in seq]

    # ------------------------------------------------------------------
    def _resolve_focal_cls(self):
        """
        FocalLoss class 찾기.
        train.py 상단에서 from Focal_loss import FocalLoss 되어 있으면 globals에서 찾음.
        """
        cls = self._focal_cls

        if cls is None:
            cls = globals().get("FocalLoss", None)

        if cls is None:
            raise RuntimeError(
                "FocalLoss를 찾을 수 없습니다. "
                "train.py 상단에 `from Focal_loss import FocalLoss`가 있는지 확인하세요."
            )

        return cls

    # ------------------------------------------------------------------
    def _build_focals(self):
        """
        task별 FocalLoss 생성.
        nn.ModuleList로 등록해두면 model.to(device) 시 함께 이동 가능.
        """
        cls = self._resolve_focal_cls()

        self.focals = nn.ModuleList([
            cls(alpha=a, gamma=self.gamma, reduction="mean")
            for a in self._alpha_list
        ])

    # ------------------------------------------------------------------
    def _run_forward(self, data, device, logger):
        """
        실제 MTMM forward 실행.
        외부에서 forward_three_modal을 주입하지 않으면 train.py의 _forward_three_modal 사용.
        """
        fwd = self._injected_fwd

        if fwd is None:
            fwd = globals().get("_forward_three_modal", None)

        if fwd is None:
            raise RuntimeError(
                "_forward_three_modal을 찾을 수 없습니다. "
                "train.py에 _forward_three_modal 함수가 정의되어 있는지 확인하세요."
            )

        return fwd(self.model, data, device, logger)

    # ------------------------------------------------------------------
    def compute_ranking_loss(self, logits, targets):
        """
        within-task ranking loss.

        positive logit이 negative logit보다 ranking_margin 이상 크도록 유도.
        positive 또는 negative가 batch에 없으면 0 loss 반환.
        """
        targets = targets.float()

        pos = logits[targets == 1]
        neg = logits[targets == 0]

        if pos.numel() == 0 or neg.numel() == 0:
            return logits.sum() * 0.0

        diff = pos.unsqueeze(1) - neg.unsqueeze(0)
        rank_loss = F.relu(self.ranking_margin - diff).mean()

        return rank_loss

    # ------------------------------------------------------------------
    def _compute_cls_loss(self, logits, targets, task_idx):
        """
        task별 classification loss 계산.
        logits: sigmoid 전 logit
        targets: 0/1 float label
        """
        targets = targets.float()

        # ------------------------------
        # Focal loss
        # ------------------------------
        if self.loss_type == "focal":
            if self.focals is None:
                raise RuntimeError(
                    "FocalLoss가 아직 초기화되지 않았습니다. "
                    "alpha='auto'인 경우 forward에서 pos_weights가 제대로 전달되는지 확인하세요."
                )
            return self.focals[task_idx](logits, targets)

        # ------------------------------
        # BCE with logits
        # ------------------------------
        target = targets

        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            # binary label smoothing
            # y=1 -> 1 - eps/2
            # y=0 -> eps/2
            target = target * (1.0 - eps) + 0.5 * eps

        pos_weight = None
        if self.use_bce_pos_weight and self._bce_pos_weight is not None:
            pos_weight = self._bce_pos_weight[task_idx].to(
                device=logits.device,
                dtype=logits.dtype,
            ).view(1)

        return F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
            reduction="mean",
        )

    # ------------------------------------------------------------------
    def forward(self, data, device, logger=None, pos_weights=None):
        """
        data:
            PyG batch. data.y는 shape (B, num_tasks) 또는 flatten된 형태여야 함.

        pos_weights:
            task별 neg/pos 비율.
            - focal alpha="auto"일 때 alpha 계산에 사용
            - BCE pos_weight를 켰을 때 pos_weight로 사용
        """

        # --------------------------------------------------------------
        # 1. focal alpha="auto" 초기화
        # --------------------------------------------------------------
        if self.loss_type == "focal" and self._alpha_auto and self.focals is None:
            if pos_weights is not None:
                self._alpha_list = alpha_from_pos_weights(
                    pos_weights.detach().cpu()
                    if torch.is_tensor(pos_weights)
                    else pos_weights
                )
            else:
                self._alpha_list = [0.5] * self.num_tasks

            self._build_focals()
            self.focals.to(device)

            if logger is not None:
                logger.info(f"[Wrapper] Focal alpha per task: {self._alpha_list}")

        # --------------------------------------------------------------
        # 2. BCE pos_weight cache
        # --------------------------------------------------------------
        if (
            self.loss_type == "bce"
            and self.use_bce_pos_weight
            and self._bce_pos_weight is None
            and pos_weights is not None
        ):
            self._bce_pos_weight = torch.as_tensor(
                pos_weights,
                dtype=torch.float32,
                device=device,
            ).reshape(-1)

            if self._bce_pos_weight.numel() != self.num_tasks:
                raise ValueError(
                    f"pos_weights length ({self._bce_pos_weight.numel()}) "
                    f"!= num_tasks ({self.num_tasks})"
                )

            if logger is not None:
                logger.info(
                    f"[Wrapper] BCE pos_weight per task: "
                    f"{self._bce_pos_weight.detach().cpu().numpy()}"
                )

        # --------------------------------------------------------------
        # 3. model forward
        # --------------------------------------------------------------
        pooled, task_outputs = self._run_forward(data, device, logger)

        # --------------------------------------------------------------
        # 4. label shape 정리
        # --------------------------------------------------------------
        y = data.y
        if torch.is_tensor(y):
            y = y.to(device)

        if y.dim() == 1:
            y = y.view(-1, self.num_tasks)

        if y.dim() != 2 or y.size(1) != self.num_tasks:
            raise ValueError(
                f"data.y shape must be (B, {self.num_tasks}), "
                f"got {tuple(y.shape)}"
            )

        # --------------------------------------------------------------
        # 5. task별 loss 계산
        # --------------------------------------------------------------
        weighted_losses = []
        active_task_weights = []
        cls_losses = []
        ranking_losses = []
        valid_counts = []

        for task_idx in range(self.num_tasks):
            logits_i = task_outputs[task_idx].squeeze(-1)

            if logits_i.dim() != 1:
                logits_i = logits_i.view(-1)

            labels_i = y[:, task_idx]

            valid = labels_i != -1

            if not valid.any():
                continue

            vl = logits_i[valid]
            vt = labels_i[valid].float()

            # classification loss
            cls_l = self._compute_cls_loss(vl, vt, task_idx)

            # ranking loss
            if self.use_ranking and self.ranking_lambda > 0.0:
                rank_l = self.compute_ranking_loss(vl, vt)
            else:
                rank_l = vl.sum() * 0.0

            task_loss = cls_l + self.ranking_lambda * rank_l

            # task uncertainty weighting
            if self.use_uncertainty:
                precision = torch.exp(-self.log_vars[task_idx])
                weighted_loss = precision * task_loss + self.log_vars[task_idx]
            else:
                weighted_loss = task_loss

            # task-specific weight
            task_w = self.task_weights[task_idx].to(
                device=weighted_loss.device,
                dtype=weighted_loss.dtype,
            )

            if not torch.isfinite(weighted_loss):
                if logger is not None:
                    logger.warning(
                        f"[Wrapper] Non-finite loss detected at task {task_idx}. "
                        "This task loss will be skipped."
                    )
                continue

            weighted_losses.append(weighted_loss * task_w)
            active_task_weights.append(task_w)

            cls_losses.append(float(cls_l.detach().cpu()))
            ranking_losses.append(float(rank_l.detach().cpu()))
            valid_counts.append(int(valid.sum().detach().cpu()))

        # --------------------------------------------------------------
        # 6. 모든 task가 -1인 batch 방어
        # --------------------------------------------------------------
        if len(weighted_losses) == 0:
            anchor = pooled if torch.is_tensor(pooled) else task_outputs[0]
            total_loss = anchor.sum() * 0.0
        else:
            stacked_losses = torch.stack(weighted_losses)
            stacked_weights = torch.stack(active_task_weights)

            if self.task_loss_reduction == "mean":
                # weight를 적용하되 전체 loss scale이 너무 커지지 않도록 weight sum으로 정규화
                total_loss = stacked_losses.sum() / stacked_weights.sum().clamp_min(1e-8)
            else:
                total_loss = stacked_losses.sum()
        # --------------------------------------------------------------
        # 7. logging용 평균 loss
        # --------------------------------------------------------------
        avg_cls = float(np.mean(cls_losses)) if cls_losses else 0.0
        avg_ranking = float(np.mean(ranking_losses)) if ranking_losses else 0.0

        # 선택적 디버그 로그
        if logger is not None and len(valid_counts) > 0:
            # 너무 자주 찍히면 로그가 많아지므로 필요할 때만 주석 해제
            # logger.debug(f"[Wrapper] valid label counts per active task: {valid_counts}")
            pass

        return total_loss, pooled, task_outputs, avg_cls, avg_ranking
# ------------------------------------------------------------------------------
# Training Function
# ------------------------------------------------------------------------------
def train(
    epoch,
    wrapper_model,
    loader,
    optimizer,
    device,
    task_type,
    metric,
    logger,
    max_grad_norm=1.0,
    pos_weights=None,
    use_amp=False,
    scaler=None,
    log_interval=5,
):
    wrapper_model.train()

    total_loss = 0.0
    total_cls = 0.0
    total_ranking = 0.0
    total_grad_norm = 0.0
    num_batches = 0
    skipped_batches = 0

    if use_amp and scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        # -----------------------------
        # Forward + loss
        # -----------------------------
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                loss, pooled, task_outputs, avg_cls, avg_ranking = wrapper_model(
                    batch,
                    device,
                    logger,
                    pos_weights=pos_weights,
                )
        else:
            loss, pooled, task_outputs, avg_cls, avg_ranking = wrapper_model(
                batch,
                device,
                logger,
                pos_weights=pos_weights,
            )

        # -----------------------------
        # NaN / Inf loss 방어
        # -----------------------------
        if not torch.isfinite(loss):
            skipped_batches += 1
            logger.warning(
                f"[Train] Non-finite loss at epoch={epoch}, batch={batch_idx}. "
                "This batch was skipped."
            )
            continue

        # -----------------------------
        # Backward
        # -----------------------------
        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                wrapper_model.parameters(),
                max_grad_norm,
            )

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                wrapper_model.parameters(),
                max_grad_norm,
            )

            optimizer.step()

        # -----------------------------
        # Logging values
        # -----------------------------
        total_loss += float(loss.detach().cpu())
        total_cls += float(avg_cls)
        total_ranking += float(avg_ranking)

        if torch.is_tensor(grad_norm):
            grad_norm_value = float(grad_norm.detach().cpu())
        else:
            grad_norm_value = float(grad_norm)

        if np.isfinite(grad_norm_value):
            total_grad_norm += grad_norm_value

        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_cls = total_cls / max(num_batches, 1)
    avg_ranking = total_ranking / max(num_batches, 1)
    avg_grad_norm = total_grad_norm / max(num_batches, 1)

    if epoch % log_interval == 0:
        loss_name = getattr(wrapper_model, "loss_type", "cls").upper()
        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Raw {loss_name}: {avg_cls:.4f} | "
            f"Ranking: {avg_ranking:.4f} | "
            f"GradNorm: {avg_grad_norm:.4f} | "
            f"Skipped: {skipped_batches}"
        )

    return avg_loss, avg_cls

# ------------------------------------------------------------------------------
# Validation Function
# ------------------------------------------------------------------------------
@torch.no_grad()
def validate(
    wrapper_model,
    val_loader,
    device,
    task_type,
    metric,
    logger,
    epoch,
    tasks=None,
    log_interval=5,
):
    wrapper_model.eval()

    num_tasks = wrapper_model.num_tasks

    if tasks is None:
        tasks = [f"task_{i}" for i in range(num_tasks)]

    opt_losses = []
    cls_losses = []
    ranking_losses = []

    y_pred_list = {i: [] for i in range(num_tasks)}
    y_label_list = {i: [] for i in range(num_tasks)}

    for batch_idx, batch in enumerate(val_loader):
        batch = batch.to(device)

        loss, pooled, task_outputs, avg_cls, avg_ranking = wrapper_model(
            batch,
            device,
            logger,
            pos_weights=None,
        )

        if torch.isfinite(loss):
            opt_losses.append(float(loss.detach().cpu()))
        else:
            logger.warning(f"[Val] Non-finite loss at batch={batch_idx}")

        cls_losses.append(float(avg_cls))
        ranking_losses.append(float(avg_ranking))

        y = batch.y

        if y.dim() == 1:
            y = y.view(-1, num_tasks)

        for i in range(num_tasks):
            logits = task_outputs[i].squeeze(-1)

            if logits.dim() != 1:
                logits = logits.view(-1)

            labels = y[:, i]
            valid_idx = labels != -1

            if not valid_idx.any():
                continue

            logits_v = logits[valid_idx]
            labels_v = labels[valid_idx].float()

            if task_type == "classification":
                scores_v = torch.sigmoid(logits_v).detach().cpu().numpy()
            else:
                scores_v = logits_v.detach().cpu().numpy()

            y_pred_list[i].extend(scores_v)
            y_label_list[i].extend(labels_v.detach().cpu().numpy())

    val_opt_loss = float(np.mean(opt_losses)) if len(opt_losses) > 0 else float("nan")
    val_cls_loss = float(np.mean(cls_losses)) if len(cls_losses) > 0 else float("nan")
    val_ranking_loss = float(np.mean(ranking_losses)) if len(ranking_losses) > 0 else float("nan")

    metric_func = get_metric_func(metric=metric)

    task_scores = {}
    finite_scores = []

    for i, task_name in enumerate(tasks):
        y_true = np.asarray(y_label_list[i], dtype=float)
        y_score = np.asarray(y_pred_list[i], dtype=float)

        if y_true.size == 0:
            task_scores[task_name] = np.nan
            continue

        try:
            score = metric_func(y_true, y_score)
        except Exception:
            score = np.nan

        task_scores[task_name] = float(score) if np.isfinite(score) else np.nan

        if np.isfinite(score):
            finite_scores.append(float(score))

    if len(finite_scores) > 0:
        avg_score = float(np.mean(finite_scores))
        min_score = float(np.min(finite_scores))

        # 평균 성능 + 가장 낮은 task 성능을 같이 반영
        val_score = avg_score
    else:
        avg_score = np.nan
        min_score = np.nan
        val_score = 0.0

    if epoch % log_interval == 0:
        loss_name = getattr(wrapper_model, "loss_type", "cls").upper()

        score_msg = " | ".join(
            [
                f"{name}:{task_scores[name]:.4f}"
                if np.isfinite(task_scores[name])
                else f"{name}:nan"
                for name in tasks
            ]
        )

        logger.info(
            f"Epoch {epoch:3d} | "
            f"Val Loss: {val_opt_loss:.4f} | "
            f"Val Raw {loss_name}: {val_cls_loss:.4f} | "
            f"Ranking: {val_ranking_loss:.4f} | "
            f"Score({metric.upper()}): {val_score:.4f} | "
            f"{score_msg}"
        )

    return val_opt_loss, val_cls_loss, val_score, task_scores

@torch.no_grad()
def test(
    model,
    criterion,
    test_loader,
    device,
    task_type="classification",
    metric="auc",
    logger=None,
    criterion_list: Optional[List[torch.nn.Module]] = None,
    thresholds=0.5,
    tasks: Optional[List[str]] = None,
    return_details: bool = False,
):
    """
    Multitask test / external validation function.

    주요 특징
    ----------
    1. criterion=None이어도 prediction-only 평가 가능
    2. -1 label 자동 mask
    3. classification에서는 sigmoid 이후 score 저장
    4. AUROC/AP 같은 threshold-free metric 계산
    5. fixed threshold 기반 Accuracy/F1/Recall/Specificity/Precision/MCC 계산
    6. test/external test에서 threshold를 새로 최적화하지 않음

    Parameters
    ----------
    model : torch.nn.Module
        학습된 base model. _forward_three_modal(model, data, device, logger)로 forward 가능해야 함.

    criterion : torch loss or None
        BCEWithLogitsLoss 등. None이면 loss 계산 생략.

    test_loader : DataLoader
        internal test 또는 external HLM/RLM test loader.

    device : torch.device

    task_type : str
        "classification" 또는 "regression".

    metric : str
        "auc", "ap", "prc" 등 get_metric_func에서 지원하는 metric.

    logger : logging.Logger or None

    criterion_list : list of torch loss or None
        task별 criterion을 따로 줄 때 사용.

    thresholds : float, list, tuple, dict
        fixed threshold.
        예:
        - 0.5
        - [0.45, 0.50, 0.40]
        - {"human": 0.45, "rat": 0.50, "mouse": 0.40}

    tasks : list of str or None
        ["human", "rat", "mouse"] 권장.
        None이면 task_0, task_1, task_2로 표시.

    return_details : bool
        False이면 기존처럼 (test_loss, avg_test_metric)만 반환.
        True이면 (test_loss, avg_test_metric, summary_df, pred_df) 반환.

    Returns
    -------
    기본:
        test_loss, avg_test_metric

    return_details=True:
        test_loss, avg_test_metric, summary_df, pred_df
    """

    model.eval()
    start = time.time()

    num_tasks = getattr(model, "num_tasks", 3)

    if tasks is None:
        tasks = [f"task_{i}" for i in range(num_tasks)]

    if len(tasks) != num_tasks:
        raise ValueError(
            f"len(tasks) must match num_tasks. "
            f"len(tasks)={len(tasks)}, num_tasks={num_tasks}"
        )

    losses = []

    y_score_list = {i: [] for i in range(num_tasks)}
    y_label_list = {i: [] for i in range(num_tasks)}

    # -------------------------------------------------------
    # threshold 가져오는 내부 함수
    # -------------------------------------------------------
    def _get_threshold(thresholds, task_idx, task_name):
        if isinstance(thresholds, dict):
            if task_name in thresholds:
                return float(thresholds[task_name])
            if task_idx in thresholds:
                return float(thresholds[task_idx])
            return 0.5

        if isinstance(thresholds, (list, tuple, np.ndarray)):
            if task_idx < len(thresholds):
                return float(thresholds[task_idx])
            return 0.5

        return float(thresholds)

    # -------------------------------------------------------
    # Prediction loop
    # -------------------------------------------------------
    for batch_idx, batch in enumerate(test_loader):
        data = batch.to(device)

        _, task_outputs = _forward_three_modal(model, data, device, logger)

        y_labels = data.y
        if y_labels.dim() == 1:
            y_labels = y_labels.view(-1, num_tasks)

        if y_labels.dim() != 2 or y_labels.size(1) != num_tasks:
            raise ValueError(
                f"data.y shape must be (B, {num_tasks}), "
                f"got {tuple(y_labels.shape)}"
            )

        for task_idx in range(num_tasks):
            logits = task_outputs[task_idx].squeeze(-1)
            if logits.dim() != 1:
                logits = logits.view(-1)

            labels = y_labels[:, task_idx]
            valid_idx = labels != -1

            if not valid_idx.any():
                continue

            logits_v = logits[valid_idx]
            labels_v = labels[valid_idx].float()

            # -----------------------------------------------
            # Loss 계산: criterion이 있을 때만 수행
            # -----------------------------------------------
            if criterion_list is not None:
                crit = criterion_list[task_idx]
            else:
                crit = criterion

            if crit is not None:
                try:
                    loss = crit(logits_v, labels_v)
                    if torch.isfinite(loss):
                        losses.append(float(loss.detach().cpu()))
                except Exception as e:
                    if logger is not None:
                        logger.warning(
                            f"[Test] Loss calculation skipped for "
                            f"{tasks[task_idx]} at batch {batch_idx}: {e}"
                        )

            # -----------------------------------------------
            # Prediction score 저장
            # -----------------------------------------------
            if task_type == "classification":
                scores_v = torch.sigmoid(logits_v).detach().cpu().numpy()
            else:
                scores_v = logits_v.detach().cpu().numpy()

            y_score_list[task_idx].extend(scores_v)
            y_label_list[task_idx].extend(labels_v.detach().cpu().numpy())

    # -------------------------------------------------------
    # Metric 계산
    # -------------------------------------------------------
    test_loss = float(np.mean(losses)) if len(losses) > 0 else float("nan")

    metric_func = get_metric_func(metric=metric)

    summary_rows = []
    pred_rows = []
    threshold_free_scores = []

    for task_idx, task_name in enumerate(tasks):
        y_true = np.asarray(y_label_list[task_idx], dtype=float)
        y_score = np.asarray(y_score_list[task_idx], dtype=float)

        if y_true.size == 0:
            if logger is not None:
                logger.info(f"[Test:{task_name}] No valid labels.")
            continue

        # threshold-free metric
        score = metric_func(y_true, y_score)
        if np.isfinite(score):
            threshold_free_scores.append(float(score))

        # fixed threshold metric
        if task_type == "classification":
            fixed_thr = _get_threshold(thresholds, task_idx, task_name)

            fixed_metrics = evaluate_with_fixed_threshold(
                y_true,
                y_score,
                threshold=fixed_thr,
            )

            summary_rows.append({
                "species": task_name,
                "n_valid": fixed_metrics["n_valid"],
                "threshold": fixed_metrics["threshold"],
                metric: float(score) if np.isfinite(score) else np.nan,
                "auc": fixed_metrics["auc"],
                "ap": fixed_metrics["ap"],
                "accuracy": fixed_metrics["accuracy"],
                "mcc": fixed_metrics["mcc"],
                "recall": fixed_metrics["recall"],
                "specificity": fixed_metrics["specificity"],
                "precision": fixed_metrics["precision"],
                "f1": fixed_metrics["f1"],
                "tn": fixed_metrics["tn"],
                "fp": fixed_metrics["fp"],
                "fn": fixed_metrics["fn"],
                "tp": fixed_metrics["tp"],
            })

            y_pred = (y_score >= fixed_thr).astype(int)

            if logger is not None:
                logger.info(
                    f"[Test:{task_name}] "
                    f"n={fixed_metrics['n_valid']} | "
                    f"thr={fixed_metrics['threshold']:.3f} | "
                    f"AUC={fixed_metrics['auc']:.4f} | "
                    f"AP={fixed_metrics['ap']:.4f} | "
                    f"ACC={fixed_metrics['accuracy']:.4f} | "
                    f"F1={fixed_metrics['f1']:.4f} | "
                    f"Recall={fixed_metrics['recall']:.4f} | "
                    f"Spec={fixed_metrics['specificity']:.4f} | "
                    f"Precision={fixed_metrics['precision']:.4f}"
                )

            for yt, ys, yp in zip(y_true, y_score, y_pred):
                pred_rows.append({
                    "species": task_name,
                    "y_true": int(yt),
                    "y_score": float(ys),
                    "y_pred": int(yp),
                    "threshold": float(fixed_thr),
                })

        else:
            summary_rows.append({
                "species": task_name,
                "n_valid": int(y_true.size),
                metric: float(score) if np.isfinite(score) else np.nan,
            })

            for yt, ys in zip(y_true, y_score):
                pred_rows.append({
                    "species": task_name,
                    "y_true": float(yt),
                    "y_score": float(ys),
                })

    avg_test_metric = (
        float(np.mean(threshold_free_scores))
        if len(threshold_free_scores) > 0
        else float("nan")
    )

    duration = time.time() - start

    if logger is not None:
        logger.info(
            f"[Test] Loss={test_loss:.4f} | "
            f"Mean {metric.upper()}={avg_test_metric:.4f} | "
            f"{duration:.2f}s"
        )
    else:
        print(
            f"[Test] Loss={test_loss:.4f} | "
            f"Mean {metric.upper()}={avg_test_metric:.4f} | "
            f"{duration:.2f}s"
        )

    summary_df = pd.DataFrame(summary_rows)
    pred_df = pd.DataFrame(pred_rows)

    if return_details:
        return test_loss, avg_test_metric, summary_df, pred_df

    return test_loss, avg_test_metric