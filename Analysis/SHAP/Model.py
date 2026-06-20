import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init, Parameter
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    global_max_pool,
)

# ============================================================
# 1. Fingerprint Encoder (MolFPEncoder)
# ============================================================

class MolFPEncoder(nn.Module):

    def __init__(
        self,
        emb_dim: int = 128,
        drop_ratio: float = 0.4,
        fp_type: str = "morgan+maccs+rdit",
        device=None,
    ):
        super().__init__()

        self.fp_type = fp_type.lower()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 각 fingerprint 차원
        morgan_dim = 2048 if "morgan" in self.fp_type else 0
        maccs_dim  = 167  if "maccs"  in self.fp_type else 0
        rdit_dim   = 2048 if "rdit"   in self.fp_type else 0  # RDKit hashed

        init_dim = morgan_dim + maccs_dim + rdit_dim
        if init_dim == 0:
            raise ValueError(f"[MolFPEncoder] fp_type='{fp_type}'가 비어 있습니다.")

        self.net = nn.Sequential(
            nn.Linear(init_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=drop_ratio),
            nn.Linear(256, emb_dim),
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
        ).to(self.device)

        # Xavier 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data) -> torch.Tensor:
        """
        data: batched torch_geometric Data
            - data.morgan_fp : (B, d1)
            - data.maccs_fp  : (B, d2)
            - data.rdit_fp   : (B, d3)
        """
        feats = []
        if "morgan" in self.fp_type:
            feats.append(data.morgan_fp.to(self.device))
        if "maccs" in self.fp_type:
            feats.append(data.maccs_fp.to(self.device))
        if "rdit" in self.fp_type:
            feats.append(data.rdit_fp.to(self.device))

        fps = torch.cat(feats, dim=1).float()   # (B, init_dim)
        return self.net(fps)                    # (B, emb_dim)


# ============================================================
# 2. Descriptor MLP (Residual MLP)
# ============================================================

def kaiming_init_(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.3,
        norm: str = "In",
        act: str = "leakyrelu",
    ):
        super().__init__()

        self.norm1 = nn.BatchNorm1d(dim) if norm == "bn" else nn.LayerNorm(dim)
        self.fc1   = nn.Linear(dim, dim)
        self.act   = nn.LeakyReLU(0.1) if act == "leakyrelu" else nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.fc2   = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h  # Residual 연결


class DescriptorMLP(nn.Module):

    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 128,
        width: int = 128,
        depth: int = 3,
        dropout: float = 0.3,
        norm: str = "In",
    ):
        super().__init__()

        self.input_norm = nn.BatchNorm1d(in_dim) if norm == "bn" else nn.LayerNorm(in_dim)

        self.stem = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.LeakyReLU(0.1),
        )

        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(
                    width,
                    dropout=dropout,
                    norm=norm,
                    act="leakyrelu",
                )
                for _ in range(depth)
            ]
        )

        self.head = nn.Sequential(
            nn.BatchNorm1d(width) if norm == "bn" else nn.LayerNorm(width),
            nn.Linear(width, emb_dim),
        )

        self.apply(kaiming_init_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.input_norm(x)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


# ============================================================
# 3. Graph Encoder (GCN → Global Max Pool)
# ============================================================
import torch
import torch.nn as nn
from torch_geometric.nn import (
    GINEConv,
    GATv2Conv,
    JumpingKnowledge,
    global_mean_pool,
    global_max_pool,
)

DEFAULT_NODE_IN_DIM = 89
DEFAULT_EDGE_DIM = 6
 
class GraphModule(nn.Module): 
    def __init__(
        self,
        out_channels: int = 256,
        hidden: int = 256,
        dropout: float = 0.4,
        device=None,
        *,
        num_layers: int = 3,
        node_in_dim: int = DEFAULT_NODE_IN_DIM,
        edge_dim: int = DEFAULT_EDGE_DIM,
        conv_type: str = "gine",
        heads: int = 4,
        jk_mode: str = "cat",
        residual: bool = True,
    ):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.conv_type = conv_type.lower()
        self.edge_dim = int(edge_dim)
        self.residual = bool(residual)
        self.num_layers = int(num_layers)
        self.jk_mode = jk_mode
 
        if self.conv_type == "gatv2" and hidden % heads != 0:
            raise ValueError(
                f"[GraphModule] gatv2: hidden({hidden}) becomes heads({heads}) "
                f"It must be divided evenly."
            )
        self.atom_encoder = nn.Linear(node_in_dim, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(self.num_layers):
            if self.conv_type == "gine":
                mlp = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.LeakyReLU(0.1),
                    nn.Linear(hidden, hidden),
                )
                conv = GINEConv(nn=mlp, train_eps=True, edge_dim=self.edge_dim)
            elif self.conv_type == "gatv2":
                conv = GATv2Conv(
                    hidden,
                    hidden // heads,
                    heads=heads,
                    edge_dim=self.edge_dim,
                    dropout=dropout,
                    add_self_loops=True,
                    fill_value=0.0,
                )  
            else:
                raise ValueError(f"[GraphModule] Unknown conv_type: {conv_type}")
 
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden))
 
        self.act = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)

        self.jk = JumpingKnowledge(
            mode=jk_mode, channels=hidden, num_layers=self.num_layers
        )
        jk_out = hidden * self.num_layers if jk_mode == "cat" else hidden
 
        self.out_proj = nn.Sequential(
            nn.Linear(2 * jk_out, out_channels),
            nn.LeakyReLU(0.1),
        )
 
        self.to(self.device)
 
    def forward(self, data) -> torch.Tensor:
        x = data.x.to(self.device).float()
        edge_index = data.edge_index.to(self.device)
        batch = data.batch.to(self.device)
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is None:
            edge_attr = torch.zeros(
                (edge_index.size(1), self.edge_dim),
                device=self.device,
                dtype=torch.float,
            )
        else:
            edge_attr = edge_attr.to(self.device).float()
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
 
        h = self.atom_encoder(x)  # (N, hidden)
 
        layer_outs = []
        for conv, bn in zip(self.convs, self.bns):
            if self.conv_type == "gine":
                m = conv(h, edge_index, edge_attr)
            else:  # gatv2
                m = conv(h, edge_index, edge_attr=edge_attr)
            m = bn(m)
            m = self.act(m)
            m = self.dropout(m)
            h = h + m if self.residual else m  
            layer_outs.append(h)
 
        # ----- JumpingKnowledge -----
        h = self.jk(layer_outs)  # (N, jk_out)
 
        # ----- mean + max readout -----
        hg = torch.cat(
            [global_mean_pool(h, batch), global_max_pool(h, batch)],
            dim=1,
        )  # (B, 2*jk_out)
 
        return self.out_proj(hg)  # (B, out_channels)

# ============================================================
# 4. Fusion Modules
# ============================================================

class WeightFusion(nn.Module):
    """
    View-wise weighted sum fusion.

    inputs:  (B, V, D)
    weight:  (V, D)
      - softmax over V with temperature
      - gate g in [0,1]: g=1 -> uniform avg, g=0 -> learned weights
    """
    def __init__(
        self,
        feat_views: int,
        feat_dim: int,
        bias: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.weight = Parameter(torch.empty(feat_views, feat_dim))
        self.bias   = Parameter(torch.empty(feat_dim)) if bias else None
        self.drop   = nn.Dropout(p=dropout)


        self.temperature: float = 1.0
        self.gate: float        = 0.0

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.weight.size(0))
            init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def set_temperature(self, t: float):
        self.temperature = float(max(t, 1e-6))

    @torch.no_grad()
    def set_gate(self, g: float):

        self.gate = float(min(max(g, 0.0), 1.0))

    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs: (B, V, D)
        returns: (B, D)
        """
        B, V, D = inputs.shape

        x = self.drop(inputs)


        w = self.weight / self.temperature               # (V, D)
        w = torch.softmax(w, dim=0)                      # (V, D)


        if self.gate != 0.0:
            uniform = inputs.new_full((V, D), 1.0 / V)
            w = (1.0 - self.gate) * w + self.gate * uniform


        out = (x * w.unsqueeze(0)).sum(dim=1)            # (B, D)

        if self.bias is not None:
            out = out + self.bias

        return out


class ConcatFusion(nn.Module):
    def __init__(
        self,
        feat_views: int,
        feat_dim: int,
        out_dim: int = None,
        norm: str = "ln",
        dropout: float = 0.5,
    ):
        super().__init__()

        self.V = int(feat_views)
        self.D = int(feat_dim)
        in_dim = self.V * self.D

        if norm == "bn":
            self.norm = nn.BatchNorm1d(in_dim)
        else:
            self.norm = nn.LayerNorm(in_dim)

        self.drop = nn.Dropout(dropout)

        if out_dim is None or out_dim == in_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (B, V, D)
        returns: (B, V*D) or (B, out_dim)
        """
        B, V, D = inputs.shape
        assert V == self.V and D == self.D, \
            f"ConcatFusion shape mismatch: got (V={V}, D={D}), expected (V={self.V}, D={self.D})"

        x = inputs.reshape(B, V * D)
        x = self.norm(x)
        x = self.drop(x)
        return self.proj(x)


# ============================================================
# 5. MTMM (FP + Graph (+ optional Desc) → Fusion → Task Heads)
# ============================================================

class MTMM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        device=None,
        num_tasks: int = 3,
        desc_in_dim: int = 256,
        # FP 관련
        fp_mode: str = "dense",                  
        fp_type: str = "morgan+maccs+rdit",      
        fp_emb_dim: int = 256,                    
        fp_out_dim: int = 256,                    
        # Graph / Fusion
        graph_out_dim: int = 256,
        fusion_dim: int = 256,
        dropout: float = 0.35,
        # Descriptor
        desc_width: int = 256,
        desc_depth: int = 3,
    ):
        super().__init__()

        self.device      = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_tasks   = int(num_tasks)
        self.desc_in_dim = int(desc_in_dim)
        self.fusion_dim  = int(fusion_dim)
        self.fp_mode     = fp_mode.lower()
        self.fp_type     = fp_type
        self.fused_norm = nn.LayerNorm(fusion_dim).to(self.device)
        self.fused_drop = nn.Dropout(dropout).to(self.device)

        # ---------------------------
        # FP encoder branch
        # ---------------------------
        if self.fp_mode == "seq":
            # 주의: forward에서 data['fp'] 또는 data['smil2vec'](LongTensor) 사용
            self.fp_encoder = FingerprintEmbed(
                vocab_size=vocab_size,
                seq_len=100,
                emb_token_dim=128,
                out_dim=fp_out_dim,
                dropout=dropout,
            ).to(self.device)
            self.fp_fc = nn.Linear(fp_out_dim, fusion_dim).to(self.device)

        elif self.fp_mode == "dense":
            self.fp_encoder = MolFPEncoder(
                emb_dim=fp_emb_dim,
                drop_ratio=dropout,
                fp_type=self.fp_type,
                device=self.device,
            ).to(self.device)
            self.fp_fc = nn.Linear(fp_emb_dim, fusion_dim).to(self.device)

        else:
            raise ValueError(f"[MTMM] Unknown fp_mode: {fp_mode}")

        # ---------------------------
        # Graph encoder
        # ---------------------------
        self.graph_encoder = GraphModule(
            out_channels=graph_out_dim,
            hidden=graph_out_dim,
            dropout=dropout,
            device=self.device,
            num_layers=3,        
            conv_type="gatv2",    # "gine" 또는 "gatv2"
            jk_mode="max",       # "cat" | "max"
            residual=True,
        )
        self.graph_fc = nn.Linear(graph_out_dim, fusion_dim).to(self.device)
        # ---------------------------
        # Descriptor MLP (optional)
        # ---------------------------
        if self.desc_in_dim > 0:
            self.desc_mlp = DescriptorMLP(
                in_dim=self.desc_in_dim,
                emb_dim=fusion_dim,
                width=desc_width,
                depth=desc_depth,
                dropout=dropout,
                norm = "ln",
            ).to(self.device)
        else:
            self.desc_mlp = None

        # ---------------------------
        # LayerNorm
        # ---------------------------
        self.graph_norm = nn.LayerNorm(fusion_dim).to(self.device)
        self.fp_norm    = nn.LayerNorm(fusion_dim).to(self.device)
        if self.desc_in_dim > 0:
            self.desc_norm = nn.LayerNorm(fusion_dim).to(self.device)
        else:
            self.desc_norm = None

        # ---------------------------
        # Fusion
        # ---------------------------
        self.n_views = 2 + (1 if self.desc_in_dim > 0 else 0)  # graph + fp (+ desc)
        self.fusion = WeightFusion(
            feat_views=self.n_views,
            feat_dim=fusion_dim,
            dropout=dropout,
        )

#         self.fusion = ConcatFusion(
#             feat_views=self.n_views,
#             feat_dim=fusion_dim,
#             out_dim=fusion_dim,
#             norm="ln",
#             dropout=dropout,
#         )

        # ---------------------------
        # Task heads
        # ---------------------------
        head_hidden = max(fusion_dim // 2, 64)

        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(self.num_tasks)
        ])

        self.outputs = nn.ModuleList([
            nn.Linear(head_hidden, 1)
            for _ in range(self.num_tasks)
        ])
    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------
    def _get_tensor(self, container, key: str):
        if isinstance(container, dict):
            return container.get(key, None)
        return getattr(container, key, None)

    def _get_fp_like(self, data):
        t = self._get_tensor(data, "fp")
        if t is None:
            t = self._get_tensor(data, "smil2vec")
        return t

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, data):
        # ----- Graph branch -----
        graph_data = self._get_tensor(data, "graph")
        if graph_data is None:
            raise RuntimeError("[MTMM] 'graph' need.")

        gfeat = self.graph_encoder(graph_data)       # (B_g, graph_out_dim)
        graph_feat = self.graph_fc(gfeat)            # (B_g, fusion_dim)
        graph_feat = self.graph_norm(graph_feat)

        # ----- FP branch -----
        if self.fp_mode == "seq":
            fp_idx = self._get_fp_like(data)        
            if fp_idx is None:
                raise RuntimeError(
                    "[MTMM] For fp_mode='seq', data['fp'] or data['smil2vec'](LongTensor) is required."
                )
            fp_idx = fp_idx.to(self.device).long()
            if torch.any(fp_idx < 0):
                raise RuntimeError(
                    f"[MTMM] fp_idx contains negative values: min={int(fp_idx.min())}"
                )
            ffeat = self.fp_encoder(fp_idx)          # (B_f, fp_out_dim)
        else:
            ffeat = self.fp_encoder(graph_data)      # (B_f, fp_emb_dim)

        fp_feat = self.fp_fc(ffeat)                  # (B_f, fusion_dim)
        fp_feat = self.fp_norm(fp_feat)

        # ----- Descriptor branch (optional) -----
        desc_feat = None
        if self.desc_mlp is not None:
            desc = self._get_tensor(data, "desc")
            if desc is not None:
                desc = desc.to(self.device).float()
                if desc.dim() == 1:
                    desc = desc.unsqueeze(1)
                elif desc.dim() > 2:
                    desc = desc.view(desc.size(0), -1)

                if desc.size(-1) != self.desc_in_dim:
                    raise RuntimeError(
                        f"[MTMM] desc_in_dim mismatch: expected {self.desc_in_dim}, got {desc.size(-1)}"
                    )

                desc_feat = self.desc_mlp(desc)      # (B_d, fusion_dim)
                if self.desc_norm is not None:
                    desc_feat = self.desc_norm(desc_feat)

        # ----- Batch align & zero-fill for missing branch -----
        sizes = [
            t.size(0)
            for t in (graph_feat, fp_feat, desc_feat)
            if t is not None
        ]
        if len(sizes) == 0:
            raise RuntimeError("[MTMM] No modality provided (graph/fp/desc are all None).")

        B = graph_feat.size(0)

        def ensure_B(x: torch.Tensor | None, name: str) -> torch.Tensor:
            if x is None:
                return torch.zeros(
                    B,
                    self.fusion_dim,
                    device=graph_feat.device,
                    dtype=graph_feat.dtype,
                )

            if x.size(0) != B:
                raise RuntimeError(
                    f"[MTMM] Batch size mismatch in {name}: "
                    f"expected {B}, got {x.size(0)}"
                )

            return x

        graph_feat = ensure_B(graph_feat, "graph")
        fp_feat = ensure_B(fp_feat, "fp")
        desc_feat = ensure_B(desc_feat, "desc")

        # ----- Fusion -----
        views = [graph_feat, fp_feat]
        if self.desc_mlp is not None:
            views.append(desc_feat)

        fusion_in = torch.stack(views, dim=1)  # (B, n_views, fusion_dim)
        fused = self.fusion(fusion_in)
        fused = self.fused_norm(fused)
        fused = self.fused_drop(fused)

        # ----- Task heads -----
        outs = []
        for head, out_lin in zip(self.task_heads, self.outputs):
            h = head(fused)
            outs.append(out_lin(h).view(-1, 1))

        return fused, tuple(outs)
