# dataset_scaffold.py
# ============================================================
# Dataset preprocessing for multi-task molecular classification
# - SMILES -> graph
# - dense fingerprints: Morgan / MACCS / RDKit
# - descriptor preprocessing with train-only scaler
# - scaffold-disjoint K-fold with fold-level descriptor scaler
# ============================================================

import os
import random
import time
import numpy as np
import pandas as pd

from typing import Optional, List, Dict, Tuple
from collections import defaultdict, Counter

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader as GeometricDataLoader

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import (
    GetMorganGenerator,
    GetRDKitFPGenerator,
)


# ============================================================
# 1. SMILES sequence encoding
# ============================================================

smi_to_seq = "(.02468@BDFHLNPRTVZ/bdfhlnprt#*%)+-/13579=ACEGIKMOSUWY[]acegimosuy\\"
seq_dict_smi = {ch: (i + 1) for i, ch in enumerate(smi_to_seq)}
MAX_SEQ_SMI_LEN = 100


def seq_smi(smile: str, max_seq_smi_len: int = MAX_SEQ_SMI_LEN) -> np.ndarray:
    idx = np.array(
        [seq_dict_smi.get(ch, 0) for ch in str(smile)[:max_seq_smi_len]],
        dtype=int,
    )

    if len(idx) < max_seq_smi_len:
        idx = np.pad(
            idx,
            (0, max_seq_smi_len - len(idx)),
            "constant",
            constant_values=0,
        )

    return idx


# ============================================================
# 2. Fingerprint generators
# ============================================================

MORGAN_GEN = GetMorganGenerator(radius=2, fpSize=2048)
RDK_GEN = GetRDKitFPGenerator(fpSize=2048)


# ============================================================
# 3. Descriptor settings
# ============================================================

USE_EXPLICIT_COLS = True

PHYS_COLS = [
    "MW", "TPSA", "iLOGP", "XLOGP3", "WLOGP", "MLOGP",
    "Silicos-IT Log P", "Consensus Log P",
    "ESOL Log S", "Ali Log S", "Silicos-IT LogSw",
    "#Heavy atoms", "#Aromatic heavy atoms", "Fraction Csp3",
    "#Rotatable bonds", "#H-bond acceptors", "#H-bond donors",
    "MR", "log Kp (cm/s)",
]

CONT_COLS = [
    "Lipinski #violations", "Ghose #violations", "Veber #violations",
    "Egan #violations", "Muegge #violations", "Bioavailability Score",
    "PAINS #alerts", "Brenk #alerts", "Leadlikeness #violations",
    "Synthetic Accessibility",
]

CAT_COLS = [
    "ESOL Class", "Ali Class", "Silicos-IT class",
    "GI absorption", "BBB permeant", "Pgp substrate",
    "CYP1A2 inhibitor", "CYP2C19 inhibitor",
    "CYP2C9 inhibitor", "CYP2D6 inhibitor", "CYP3A4 inhibitor",
]

BINARY_POSITIVE_MAP = {
    "GI absorption": "High",
    "BBB permeant": "Yes",
    "Pgp substrate": "Yes",
    "CYP1A2 inhibitor": "Yes",
    "CYP2C19 inhibitor": "Yes",
    "CYP2C9 inhibitor": "Yes",
    "CYP2D6 inhibitor": "Yes",
    "CYP3A4 inhibitor": "Yes",
}


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def _make_desc_df_raw(df_: pd.DataFrame, tasks: List[str]) -> pd.DataFrame:
    """
    Build the descriptor subset in raw form.
    No NaN/Inf imputation or scaling is done here.
    """

    exclude = set(
        [
            "Cano_Smile",
            "SMILES",
            "PUBCHEM_EXT_DATASOURCE_SMILES",
            "_canonical_smiles",
        ]
        + list(tasks)
    )

    if USE_EXPLICIT_COLS:
        phys_in = [c for c in PHYS_COLS if c in df_.columns and c not in exclude]
        cont_in = [c for c in CONT_COLS if c in df_.columns and c not in exclude]
        cat_in = [c for c in CAT_COLS if c in df_.columns and c not in exclude]

        # numeric descriptors
        if phys_in or cont_in:
            num_df = df_[phys_in + cont_in].copy()
            for c in num_df.columns:
                num_df[c] = _coerce_numeric_series(num_df[c])
        else:
            num_df = pd.DataFrame(index=df_.index)

        # categorical descriptors
        if cat_in:
            cat_cols = []

            for c in cat_in:
                col = df_[c].astype(str).str.strip()
                col = col.replace({"nan": np.nan, "None": np.nan, "": np.nan})

                uniq = sorted([u for u in col.dropna().unique()])

                if 0 < len(uniq) <= 2:
                    positive = BINARY_POSITIVE_MAP.get(c, uniq[-1])
                    bin_series = (col == positive).astype(float)
                    bin_series = bin_series.fillna(0.0)
                    bin_series.name = f"{c}__{positive}"
                    cat_cols.append(bin_series)
                else:
                    dummies = pd.get_dummies(col, prefix=c, dummy_na=False)
                    cat_cols.append(dummies)

            cat_df = pd.concat(cat_cols, axis=1)
        else:
            cat_df = pd.DataFrame(index=df_.index)

        desc_df = pd.concat([num_df, cat_df], axis=1)

    else:
        candidate_numeric_cols = []

        for c in df_.columns:
            if c in exclude:
                continue

            coerced = _coerce_numeric_series(df_[c])
            if coerced.notna().mean() > 0.9:
                candidate_numeric_cols.append(c)

        if candidate_numeric_cols:
            desc_df = df_[candidate_numeric_cols].copy()
            for c in desc_df.columns:
                desc_df[c] = _coerce_numeric_series(desc_df[c])
        else:
            desc_df = pd.DataFrame(index=df_.index)

    if desc_df.shape[1] == 0:
        desc_df = pd.DataFrame(
            {"__desc_dummy__": np.zeros(len(df_), dtype=float)},
            index=df_.index,
        )

    return desc_df


def build_desc_df_scaled(
    df: pd.DataFrame,
    tasks: List[str],
    logger=None,
    fixed_cols: Optional[List[str]] = None,
    scaler: Optional[StandardScaler] = None,
    impute_values: Optional[Dict[str, float]] = None,
    add_missing_indicators: bool = True,
):
    """
    Descriptor preprocessing.

    Train:
        fixed_cols=None, scaler=None, impute_values=None
        -> build columns, compute medians, fit StandardScaler

    Val/Test/External:
        fixed_cols=train_cols, scaler=train_scaler, impute_values=train_impute
        -> reindex to train schema, impute with train medians, transform with train scaler
    """

    desc_df_raw = _make_desc_df_raw(df, tasks)

    numeric_base_cols = [
        c for c in desc_df_raw.columns
        if c in set(PHYS_COLS + CONT_COLS)
    ]

    desc_df_proc = desc_df_raw.replace([np.inf, -np.inf], np.nan)

    if add_missing_indicators:
        for c in numeric_base_cols:
            miss_col = f"{c}__missing"
            desc_df_proc[miss_col] = desc_df_proc[c].isna().astype(float)

    if fixed_cols is None:
        cols = list(desc_df_proc.columns)
    else:
        for c in fixed_cols:
            if c not in desc_df_proc.columns:
                desc_df_proc[c] = 0.0

        desc_df_proc = desc_df_proc.reindex(columns=fixed_cols)
        cols = list(fixed_cols)

    if impute_values is None:
        impute_values = {}

        for c in cols:
            med = desc_df_proc[c].median()

            if pd.isna(med):
                med = 0.0

            impute_values[c] = float(med)
    else:
        impute_values = dict(impute_values)

    for c in cols:
        desc_df_proc[c] = desc_df_proc[c].fillna(impute_values.get(c, 0.0))

    X = desc_df_proc.values.astype("float32")

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    desc_df_scaled = pd.DataFrame(
        X_scaled,
        columns=cols,
        index=df.index,
    )

    if logger:
        n_missing_indicators = sum(c.endswith("__missing") for c in cols)
        logger.info(
            f"[Descriptor] cols={desc_df_scaled.shape[1]} | "
            f"missing_indicators={n_missing_indicators} | "
            f"fit_scaler={scaler is not None and fixed_cols is None}"
        )

    return desc_df_scaled, cols, scaler, impute_values


# ============================================================
# 4. Graph feature encoding
# ============================================================

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def _explicit_valence(atom):
    if hasattr(Chem, "ValenceType"):
        return atom.GetValence(Chem.ValenceType.EXPLICIT)
    return atom.GetExplicitValence()


# NOTE:
# To stay compatible with DEFAULT_NODE_IN_DIM=89 in model.py,
# the atom list dim is kept; if you shrink the element list here, update model.py too.
ATOM_LIST = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg",
    "Na", "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl",
    "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H",
    "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr",
    "Pt", "Hg", "Pb", "Nd", "Ru", "W", "Unknown", "Mo", "Sr",
    "Bi", "S", "Ba", "Be", "Dy",
]

DEGREE_LIST = [0, 1, 2, 3, 4, 5, 6]
FORMAL_CHARGE_LIST = [-1, 0, 1]
EXPLICIT_VALENCE_LIST = [0, 1, 2, 3, 4, 5, 6]
NUM_H_LIST = [0, 1, 2, 3, 4, 5]
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    "UNSPECIFIED",
    "S",
]
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]


def atom_features(atom):
    features = (
        one_of_k_encoding_unk(atom.GetSymbol(), ATOM_LIST)
        + one_of_k_encoding_unk(atom.GetDegree(), DEGREE_LIST)
        + one_of_k_encoding_unk(atom.GetFormalCharge(), FORMAL_CHARGE_LIST)
        + one_of_k_encoding_unk(_explicit_valence(atom), EXPLICIT_VALENCE_LIST)
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), NUM_H_LIST)
        + one_of_k_encoding_unk(atom.GetHybridization(), HYBRIDIZATION_LIST)
        + [atom.GetIsAromatic()]
        + one_of_k_encoding_unk(atom.GetChiralTag(), CHIRALITY_LIST)
    )

    return np.array(features, dtype=float)


def bond_features(bond):
    bt = bond.GetBondType()

    return np.array(
        [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing(),
        ],
        dtype=float,
    )


# ============================================================
# 5. CSV and Data conversion helpers
# ============================================================

SMI_CANDIDATES = ["Cano_Smile", "SMILES", "PUBCHEM_EXT_DATASOURCE_SMILES"]


def find_smiles_column(df: pd.DataFrame) -> str:
    smi_col = next((c for c in SMI_CANDIDATES if c in df.columns), None)

    if smi_col is None:
        raise ValueError(f"Input CSV must contain one of {SMI_CANDIDATES}")

    return smi_col


def prepare_dataframe(
    dataset_path: str,
    tasks: List[str],
    logger=None,
) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(dataset_path)
    smi_col = find_smiles_column(df)

    for t in tasks:
        if t not in df.columns:
            df[t] = -1
        df[t] = _coerce_numeric_series(df[t]).fillna(-1).astype(np.float32)

    label_mat = df[tasks].values
    valid_label_mask = (label_mat != -1).any(axis=1)
    df = df[valid_label_mask].reset_index(drop=True)

    canonical_smiles = []
    keep_idx = []
    invalid_smiles = 0

    for idx, smi in enumerate(df[smi_col].astype(str)):
        mol = Chem.MolFromSmiles(smi.strip())

        if mol is None:
            invalid_smiles += 1
            continue

        canonical_smiles.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        keep_idx.append(idx)

    df = df.iloc[keep_idx].reset_index(drop=True)
    df["_canonical_smiles"] = canonical_smiles

    if logger:
        logger.info(
            f"[prepare_dataframe] {os.path.basename(dataset_path)} | "
            f"kept={len(df)} | invalid_smiles={invalid_smiles}"
        )

    return df, "_canonical_smiles"


def dataframe_to_data_list(
    df: pd.DataFrame,
    tasks: List[str],
    desc_df: pd.DataFrame,
    smi_col: str,
    logger=None,
):
    data_list = []
    smiles_attr = []
    invalid_smiles = 0

    for i, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Graph Conversion",
    ):
        smi = str(row[smi_col]).strip()
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            invalid_smiles += 1
            continue

        canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=True)

        atom_feat = [atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(np.array(atom_feat), dtype=torch.float)

        edge_indices = []
        edge_attrs = []

        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            b_feat = bond_features(bond)

            edge_indices.append([start, end])
            edge_attrs.append(b_feat)
            edge_indices.append([end, start])
            edge_attrs.append(b_feat)

        if len(edge_indices) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 6), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.smiles = canonical_smi

        # Morgan / RDKit hashed fingerprints
        for gen, name in [(MORGAN_GEN, "morgan_fp"), (RDK_GEN, "rdit_fp")]:
            bv = gen.GetFingerprint(mol)
            arr = np.zeros((bv.GetNumBits(),), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bv, arr)
            setattr(
                data,
                name,
                torch.from_numpy(arr.astype(np.float32)).unsqueeze(0),
            )

        # MACCS
        bv_maccs = MACCSkeys.GenMACCSKeys(mol)
        arr_maccs = np.zeros((bv_maccs.GetNumBits(),), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bv_maccs, arr_maccs)
        data.maccs_fp = torch.from_numpy(arr_maccs.astype(np.float32)).unsqueeze(0)

        data.y = torch.tensor(
            row[tasks].values.astype(np.float32),
            dtype=torch.float,
        ).unsqueeze(0)

        data.desc = torch.tensor(
            desc_df.iloc[i].values.astype(np.float32),
            dtype=torch.float,
        ).unsqueeze(0)

        data.smil2vec = torch.LongTensor(seq_smi(canonical_smi)).unsqueeze(0)

        data_list.append(data)
        smiles_attr.append(canonical_smi)

    if logger:
        logger.info(
            f"[dataframe_to_data_list] valid={len(data_list)} | "
            f"invalid_smiles={invalid_smiles}"
        )

    return data_list, smiles_attr


# ============================================================
# 6. Safe torch load
# ============================================================

def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# ============================================================
# 7. MolDataset
# ============================================================

class MolDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        dataset,
        task_type,
        tasks,
        logger=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        desc_cols=None,
        desc_scaler=None,
        desc_impute_values=None,
        processed_suffix="default",
    ):
        self.tasks = tasks
        self.dataset = dataset
        self.task_type = task_type
        self.logger = logger

        self.fixed_desc_cols = desc_cols
        self.fixed_desc_scaler = desc_scaler
        self.fixed_desc_impute_values = desc_impute_values
        self.processed_suffix = processed_suffix

        super().__init__(root, transform, pre_transform, pre_filter)

        loaded = _safe_torch_load(self.processed_paths[0])

        if isinstance(loaded, tuple) and len(loaded) == 6:
            (
                self.data,
                self.slices,
                self.smiles_list,
                self.desc_cols_,
                self.desc_scaler_,
                self.desc_impute_values_,
            ) = loaded

        elif isinstance(loaded, tuple) and len(loaded) == 5:
            (
                self.data,
                self.slices,
                self.smiles_list,
                self.desc_cols_,
                self.desc_scaler_,
            ) = loaded
            self.desc_impute_values_ = None

        else:
            self.data, self.slices, self.smiles_list = loaded[:3]
            self.desc_cols_ = None
            self.desc_scaler_ = None
            self.desc_impute_values_ = None

    @property
    def raw_file_names(self):
        return [self.dataset]

    @property
    def processed_file_names(self):
        base = os.path.splitext(os.path.basename(self.dataset))[0]
        return [f"{base}_{self.processed_suffix}.pt"]

    def process(self):
        dataset_path = os.path.join(self.root, self.dataset)

        df, smi_col = prepare_dataframe(
            dataset_path=dataset_path,
            tasks=self.tasks,
            logger=self.logger,
        )

        desc_df, used_cols, scaler, impute_values = build_desc_df_scaled(
            df,
            tasks=self.tasks,
            logger=self.logger,
            fixed_cols=self.fixed_desc_cols,
            scaler=self.fixed_desc_scaler,
            impute_values=self.fixed_desc_impute_values,
        )

        self.desc_cols_ = list(used_cols)
        self.desc_scaler_ = scaler
        self.desc_impute_values_ = impute_values

        data_list, smiles_attr = dataframe_to_data_list(
            df=df,
            tasks=self.tasks,
            desc_df=desc_df,
            smi_col=smi_col,
            logger=self.logger,
        )

        if len(data_list) == 0:
            raise RuntimeError(f"[MolDataset] No valid molecules found in {self.dataset}")

        data, slices = self.collate(data_list)

        torch.save(
            (
                data,
                slices,
                smiles_attr,
                self.desc_cols_,
                self.desc_scaler_,
                self.desc_impute_values_,
            ),
            self.processed_paths[0],
        )


# ============================================================
# 8. Scaffold split utilities
# ============================================================

def _murcko_scaffold(smi: str, include_chirality: bool = False) -> str:
    try:
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            return ""

        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=include_chirality,
        ) or ""

    except Exception:
        return ""


def _group_indices_by_scaffold(
    smiles_list: List[str],
    include_chirality: bool = False,
):
    buckets = defaultdict(list)

    for i, smi in enumerate(smiles_list):
        scaf = _murcko_scaffold(smi, include_chirality=include_chirality)
        key = scaf if len(scaf) > 0 else smi
        buckets[key].append(i)

    return buckets


def _greedy_pack_scaffolds_to_folds(
    scaffold_buckets: Dict[str, List[int]],
    n_splits: int,
    seed: int = 2026,
):
    groups = list(scaffold_buckets.items())
    groups.sort(key=lambda kv: (-len(kv[1]), kv[0]))

    fold_bins = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    for _, idx_list in groups:
        k = min(range(n_splits), key=lambda f: fold_sizes[f])
        fold_bins[k].extend(idx_list)
        fold_sizes[k] += len(idx_list)

    return fold_bins


def _log_fold_stats(data_list, tasks, logger, tag=""):
    if not logger:
        return

    cnt = Counter()
    pos_cnt = Counter()
    neg_cnt = Counter()

    for d in data_list:
        yrow = d.y.view(-1)

        for i, t in enumerate(tasks):
            val = yrow[i].item()

            if val != -1:
                cnt[t] += 1

                if val == 1:
                    pos_cnt[t] += 1
                elif val == 0:
                    neg_cnt[t] += 1

    logger.info(
        f"{tag} size={len(data_list)} | "
        f"available={dict(cnt)} | "
        f"pos={dict(pos_cnt)} | "
        f"neg={dict(neg_cnt)}"
    )


# ============================================================
# 9. Scaffold K-fold loader with fold-level descriptor scaler
# ============================================================
import hashlib
import json


def _safe_torch_load_cache(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _make_fold_cache_path(
    data_path,
    dataset_name,
    tasks,
    n_splits,
    include_chirality,
    seed,
    cache_dir=None,
    cache_version="v1",
):
    """
    Build the per-fold graph/data_list cache path.
    Includes dataset file size/mtime, so a changed CSV automatically uses a different cache.
    """

    dataset_path = os.path.join(data_path, dataset_name)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(dataset_path)

    stat = os.stat(dataset_path)

    meta = {
        "dataset_name": dataset_name,
        "tasks": list(tasks),
        "n_splits": int(n_splits),
        "include_chirality": bool(include_chirality),
        "seed": int(seed),
        "file_size": int(stat.st_size),
        "file_mtime": int(stat.st_mtime),
        "cache_version": str(cache_version),
    }

    key = hashlib.md5(
        json.dumps(meta, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    if cache_dir is None:
        cache_dir = os.path.join(data_path, "fold_graph_cache")

    os.makedirs(cache_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(dataset_name))[0]
    cache_path = os.path.join(
        cache_dir,
        f"{base}_scaffold{n_splits}_seed{seed}_{key}.pt",
    )

    return cache_path, meta


def _build_loaders_from_cached_fold_lists(
    cached_train_lists,
    cached_val_lists,
    batch_size,
):
    train_loaders = []
    val_loaders = []

    for train_data_list, val_data_list in zip(cached_train_lists, cached_val_lists):
        train_loader = GeometricDataLoader(
            train_data_list,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        val_loader = GeometricDataLoader(
            val_data_list,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders
def build_scaffold_kfold_loader(
    data_path: str,
    dataset_name: str,
    task_type: str,
    batch_size: int,
    tasks: List[str],
    logger=None,
    n_splits: int = 10,
    include_chirality: bool = False,
    seed: int = 2026,
    use_cache: bool = True,
    force_rebuild: bool = False,
    cache_dir: str = None,
    cache_version: str = "v1",
):
    """
    Scaffold K-fold loader with fold-level descriptor scaler + fold graph cache.

    First run:
        raw CSV -> scaffold split -> per-fold descriptor scaler fit/transform
        -> graph conversion -> save fold cache

    Subsequent runs:
        load fold cache -> build DataLoader

    Returns
    -------
    train_loaders : list[DataLoader]
    val_loaders   : list[DataLoader]
    fold_desc_info: list[dict]
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --------------------------------------------------------
    # 0. build cache path
    # --------------------------------------------------------
    cache_path, cache_meta = _make_fold_cache_path(
        data_path=data_path,
        dataset_name=dataset_name,
        tasks=tasks,
        n_splits=n_splits,
        include_chirality=include_chirality,
        seed=seed,
        cache_dir=cache_dir,
        cache_version=cache_version,
    )

    # --------------------------------------------------------
    # 1. Cache load
    # --------------------------------------------------------
    if use_cache and (not force_rebuild) and os.path.exists(cache_path):
        if logger:
            logger.info("=" * 70)
            logger.info(f"[Scaffold Cache] Loading cached folds:")
            logger.info(f"  {cache_path}")
            logger.info("=" * 70)

        payload = _safe_torch_load_cache(cache_path)

        cached_train_lists = payload["train_data_lists"]
        cached_val_lists = payload["val_data_lists"]
        fold_desc_info = payload["fold_desc_info"]

        train_loaders, val_loaders = _build_loaders_from_cached_fold_lists(
            cached_train_lists=cached_train_lists,
            cached_val_lists=cached_val_lists,
            batch_size=batch_size,
        )

        if logger:
            logger.info(
                f"[Scaffold Cache] Loaded {len(train_loaders)} folds from cache."
            )

            for fold_idx, (tr_list, va_list) in enumerate(
                zip(cached_train_lists, cached_val_lists)
            ):
                logger.info(
                    f"[Cached Fold {fold_idx + 1}/{len(train_loaders)}] "
                    f"Train={len(tr_list)}, Val={len(va_list)}"
                )
                _log_fold_stats(tr_list, tasks, logger, tag="  Train")
                _log_fold_stats(va_list, tasks, logger, tag="  Val  ")

        return train_loaders, val_loaders, fold_desc_info

    # --------------------------------------------------------
    # 2. build fresh if no cache
    # --------------------------------------------------------
    if logger:
        logger.info("=" * 70)
        logger.info("[Scaffold Cache] Cache not found or force_rebuild=True.")
        logger.info("[Scaffold Cache] Building folds from raw CSV.")
        logger.info(f"[Scaffold Cache] Cache will be saved to: {cache_path}")
        logger.info("=" * 70)

    dataset_path = os.path.join(data_path, dataset_name)

    df, smi_col = prepare_dataframe(
        dataset_path=dataset_path,
        tasks=tasks,
        logger=logger,
    )

    smiles_list = df[smi_col].tolist()

    buckets = _group_indices_by_scaffold(
        smiles_list,
        include_chirality=include_chirality,
    )

    fold_bins = _greedy_pack_scaffolds_to_folds(
        buckets,
        n_splits=n_splits,
        seed=seed,
    )

    train_loaders = []
    val_loaders = []
    fold_desc_info = []

    cached_train_lists = []
    cached_val_lists = []

    # --------------------------------------------------------
    # 3. build per-fold data_list
    # --------------------------------------------------------
    for fold_idx in range(n_splits):
        val_index = sorted(fold_bins[fold_idx])
        train_index = sorted(
            [
                i
                for k in range(n_splits)
                if k != fold_idx
                for i in fold_bins[k]
            ]
        )

        train_df = df.iloc[train_index].reset_index(drop=True)
        val_df = df.iloc[val_index].reset_index(drop=True)

        # fit descriptor scaler on fold-train only
        train_desc_df, desc_cols, desc_scaler, desc_impute_values = build_desc_df_scaled(
            train_df,
            tasks=tasks,
            logger=logger,
            fixed_cols=None,
            scaler=None,
            impute_values=None,
        )

        # transform fold-val with the train scaler
        val_desc_df, _, _, _ = build_desc_df_scaled(
            val_df,
            tasks=tasks,
            logger=logger,
            fixed_cols=desc_cols,
            scaler=desc_scaler,
            impute_values=desc_impute_values,
        )

        train_data_list, _ = dataframe_to_data_list(
            df=train_df,
            tasks=tasks,
            desc_df=train_desc_df,
            smi_col=smi_col,
            logger=logger,
        )

        val_data_list, _ = dataframe_to_data_list(
            df=val_df,
            tasks=tasks,
            desc_df=val_desc_df,
            smi_col=smi_col,
            logger=logger,
        )

        train_loader = GeometricDataLoader(
            train_data_list,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        val_loader = GeometricDataLoader(
            val_data_list,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        if logger:
            logger.info(
                f"[Scaffold Fold {fold_idx + 1}/{n_splits}] "
                f"Train={len(train_data_list)}, Val={len(val_data_list)}"
            )
            _log_fold_stats(train_data_list, tasks, logger, tag="  Train")
            _log_fold_stats(val_data_list, tasks, logger, tag="  Val  ")

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

        cached_train_lists.append(train_data_list)
        cached_val_lists.append(val_data_list)

        fold_desc_info.append(
            {
                "desc_cols": desc_cols,
                "desc_scaler": desc_scaler,
                "desc_impute_values": desc_impute_values,
                "train_index": train_index,
                "val_index": val_index,
            }
        )

    # --------------------------------------------------------
    # 4. Save cache
    # --------------------------------------------------------
    if use_cache:
        payload = {
            "meta": cache_meta,
            "train_data_lists": cached_train_lists,
            "val_data_lists": cached_val_lists,
            "fold_desc_info": fold_desc_info,
        }

        torch.save(payload, cache_path)

        if logger:
            logger.info("=" * 70)
            logger.info(f"[Scaffold Cache] Saved fold cache:")
            logger.info(f"  {cache_path}")
            logger.info("=" * 70)

    return train_loaders, val_loaders, fold_desc_info
# ============================================================
# 10. Standard train/val/test loader
# ============================================================

def build_loader(
    data_path,
    dataset_names,
    task_type,
    batch_size,
    tasks,
    logger=None,
):
    train_loader = None
    val_loader = None
    test_loader = None

    train_desc_cols = None
    train_desc_scaler = None
    train_desc_impute_values = None

    if "train" in dataset_names:
        train_dataset = MolDataset(
            root=data_path,
            dataset=dataset_names["train"],
            task_type=task_type,
            tasks=tasks,
            logger=logger,
            desc_cols=None,
            desc_scaler=None,
            desc_impute_values=None,
            processed_suffix="train_fit",
        )

        train_desc_cols = getattr(train_dataset, "desc_cols_", None)
        train_desc_scaler = getattr(train_dataset, "desc_scaler_", None)
        train_desc_impute_values = getattr(train_dataset, "desc_impute_values_", None)

        train_loader = GeometricDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    if "val" in dataset_names:
        val_dataset = MolDataset(
            root=data_path,
            dataset=dataset_names["val"],
            task_type=task_type,
            tasks=tasks,
            logger=logger,
            desc_cols=train_desc_cols,
            desc_scaler=train_desc_scaler,
            desc_impute_values=train_desc_impute_values,
            processed_suffix="val_using_train_scaler",
        )

        val_loader = GeometricDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

    if "test" in dataset_names:
        test_dataset = MolDataset(
            root=data_path,
            dataset=dataset_names["test"],
            task_type=task_type,
            tasks=tasks,
            logger=logger,
            desc_cols=train_desc_cols,
            desc_scaler=train_desc_scaler,
            desc_impute_values=train_desc_impute_values,
            processed_suffix="test_using_train_scaler",
        )

        test_loader = GeometricDataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_desc_cols,
        train_desc_scaler,
        train_desc_impute_values,
    )


# ============================================================
# 11. External loader
# ============================================================

def build_external_loader(
    data_path,
    dataset_name,
    task_type,
    batch_size,
    tasks,
    logger,
    train_desc_cols,
    train_desc_scaler,
    train_desc_impute_values,
    processed_suffix="external_using_train_scaler",
):
    """
    External HLM/RLM test loader.

    You must pass the train descriptor schema/scaler/impute_values.
    """

    external_dataset = MolDataset(
        root=data_path,
        dataset=dataset_name,
        task_type=task_type,
        tasks=tasks,
        logger=logger,
        desc_cols=train_desc_cols,
        desc_scaler=train_desc_scaler,
        desc_impute_values=train_desc_impute_values,
        processed_suffix=processed_suffix,
    )

    external_loader = GeometricDataLoader(
        external_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return external_loader


# ============================================================
# 12. Quick feature dimension check
# ============================================================

def check_atom_feature_dim(smiles: str = "CCO") -> int:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return len(atom_features(mol.GetAtomWithIdx(0)))
