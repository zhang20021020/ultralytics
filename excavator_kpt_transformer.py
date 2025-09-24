# excavator_kpt_transformer.py
import math
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 几何特征构造（可选但强烈建议）
# -----------------------------
def build_geo_features(kpts: torch.Tensor) -> torch.Tensor:
    """
    输入:  kpts: (B, T, K, 3)  其中最后一维为 (x, y, v)
    输出:  geo:  (B, T, G)    几何特征，G为特征维数
    说明:  下方按你的示意点位做了几种常用几何量：
      - 铲斗尖端(1) / 铲斗-动臂铰点(2) / 动臂中段(3) / 动臂根部(4)
      - 驾驶室参考点(5) / 底盘中心(6) / 左前轮(7) / 右后轮(9)
    如你的关键点定义不同，请自行调整索引。
    """
    # 索引换成 0-based
    idx = {
        "bucket_tip": 0,   # 点1
        "bucket_hinge": 1, # 点2
        "boom_mid": 2,     # 点3
        "boom_root": 3,    # 点4
        "cabin": 4,        # 点5
        "chassis": 5,      # 点6
        "wheel_l": 6,      # 点7
        "wheel_r": 8,      # 点9
    }

    x = kpts[..., 0]  # (B, T, K)
    y = kpts[..., 1]  # (B, T, K)
    v = kpts[..., 2]  # (B, T, K)

    def _vec(a: int, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return x[..., b] - x[..., a], y[..., b] - y[..., a]

    # 动臂向量: root->mid，mid->hinge，铲斗向量: hinge->tip
    vx_rm, vy_rm = _vec(idx["boom_root"], idx["boom_mid"])
    vx_mh, vy_mh = _vec(idx["boom_mid"], idx["bucket_hinge"])
    vx_ht, vy_ht = _vec(idx["bucket_hinge"], idx["bucket_tip"])

    # 与水平线夹角（-pi~pi）
    def angle(vx, vy):
        return torch.atan2(vy, vx)

    a_rm = angle(vx_rm, vy_rm)      # 根到中段
    a_mh = angle(vx_mh, vy_mh)      # 中段到铰点
    a_ht = angle(vx_ht, vy_ht)      # 铰点到铲尖（铲斗角）

    # 段长（做尺度归一化：除以两轮中心距）
    # 轮距近似: 左前轮(7) <-> 右后轮(9) 的距离（如为履带车，可换成两端接触点）
    dx_wr = x[..., idx["wheel_r"]] - x[..., idx["wheel_l"]]
    dy_wr = y[..., idx["wheel_r"]] - y[..., idx["wheel_l"]]
    wheel_span = torch.sqrt(dx_wr**2 + dy_wr**2).clamp_min(1e-4)

    def seg_len(vx, vy):
        return torch.sqrt(vx**2 + vy**2) / wheel_span

    len_rm = seg_len(vx_rm, vy_rm)
    len_mh = seg_len(vx_mh, vy_mh)
    len_ht = seg_len(vx_ht, vy_ht)

    # 铲斗尖端相对底盘中心的高度差（y 轴方向，y越大代表越靠下的话请取反）
    rel_h_tip = (y[..., idx["chassis"]] - y[..., idx["bucket_tip"]]) / wheel_span

    # 可见性比例（保证遮挡时模型鲁棒）
    vis_ratio = (v > 0).float().mean(dim=-1)  # (B, T)

    # 拼接几何特征
    geo = torch.stack([
        a_rm, a_mh, a_ht,
        len_rm, len_mh, len_ht,
        rel_h_tip, vis_ratio
    ], dim=-1)  # (B, T, 8)

    # 角度归一化到 [-1, 1]
    geo[..., 0:3] = geo[..., 0:3] / math.pi
    return geo


# -----------------------------
# 可学习位置编码
# -----------------------------
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:, :T, :]


# -----------------------------
# 主模型
# -----------------------------
class ExcavatorKPTTransformer(nn.Module):
    def __init__(
        self,
        num_keypoints: int = 9,
        use_visibility: bool = True,
        use_geo: bool = True,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 2,
        use_cls_token: bool = True,
        max_len: int = 1024
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.use_visibility = use_visibility
        self.use_geo = use_geo
        self.use_cls_token = use_cls_token

        base_feat = 2 if not use_visibility else 3  # (x, y[, v])
        frame_in_dim = num_keypoints * base_feat

        # 帧内线性投影
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(frame_in_dim),
            nn.Linear(frame_in_dim, d_model)
        )

        # 可选几何特征分支
        geo_dim = 8 if use_geo else 0
        if use_geo:
            self.geo_proj = nn.Sequential(
                nn.LayerNorm(geo_dim),
                nn.Linear(geo_dim, d_model)
            )

        # [CLS] token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = LearnablePositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, kpts: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        kpts:    (B, T, K, 3)  -> (x, y, v)  其中 v 可选
        lengths: (B,) 每个样本的有效帧长。若提供，将做 padding mask。
        返回:    (B, num_classes) logits
        """
        B, T, K, C = kpts.shape
        assert K == self.num_keypoints, "关键点数量不匹配"

        # 选择是否使用 v
        if not self.use_visibility and C == 3:
            kpts = kpts[..., :2]  # 丢弃 v
            C = 2

        frame_feat = kpts.reshape(B, T, K * C)  # (B, T, K*C)
        f_emb = self.frame_proj(frame_feat)     # (B, T, d)

        if self.use_geo:
            geo = build_geo_features(kpts)         # (B, T, 8)
            g_emb = self.geo_proj(geo)             # (B, T, d)
            x = f_emb + g_emb                      # 融合
        else:
            x = f_emb

        # 可选 [CLS]
        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, -1)  # (B,1,d)
            x = torch.cat([cls, x], dim=1)         # (B, T+1, d)

        # 位置编码
        x = self.pos_enc(x)

        # padding mask: True 表示需要mask（无效位置）
        src_key_padding_mask = None
        if lengths is not None:
            pad = x.new_ones((B, x.size(1)), dtype=torch.bool)  # (B, T[+1])
            offset = 1 if self.use_cls_token else 0
            for i, L in enumerate(lengths.tolist()):
                pad[i, offset:offset+L] = False
            # 其余位置为 True（mask）
            src_key_padding_mask = pad

        z = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T[+1], d)

        if self.use_cls_token:
            pooled = z[:, 0]  # [CLS]
        else:
            # 对有效帧平均池化
            if lengths is None:
                pooled = z.mean(dim=1)
            else:
                mask = (~src_key_padding_mask).float()
                if self.use_cls_token:
                    mask = mask[:, 1:]  # 去掉CLS
                    feats = z[:, 1:, :]
                else:
                    feats = z
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
                pooled = (feats * mask.unsqueeze(-1)).sum(dim=1) / denom

        logits = self.head(pooled)
        return logits
