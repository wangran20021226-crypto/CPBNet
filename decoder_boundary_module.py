
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Dict



_SDF_N_WORKERS: int = min(8, os.cpu_count() or 4)
_SDF_POOL: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=_SDF_N_WORKERS)


def _sdf_one_image(args: Tuple[np.ndarray, int]) -> np.ndarray:
    """
    单张图的多类别 SDF 计算（线程安全）。
    scipy.ndimage.distance_transform_edt 无全局状态，可安全并发调用。

    BUG A 修复（v8.2）
    ──────────────────
    原版对每个类别的 SDF 做了 `sdf_c / (m + 1e-6)` 归一化到 [-1, 1]，
    与 compute_soft_boundary_from_sdf 中 sigma=3.0 配合后：
      - 256x256 ACDC 图像: 1 物理像素 ≈ 0.01-0.03 SDF 单位
      - soft_boundary = exp(-0.01/3) ≈ 0.997，所有像素都接近 1
    导致：
      - 分层采样退化（is_bnd > 0.5 几乎处处为真）
      - BTNA 的 w_bnd_eff 全图都接近 1，边界正则失去聚焦
      - 双温度机制失效（sb_k 永远接近 1）

    修复：去掉归一化，让 SDF 保持原始像素距离单位；sigma=3.0 对应
          约 3 像素宽的边界带，与设计意图一致。
    特殊值（类别完全不存在 / 完全占满）改用 LARGE 代替 ±1，
    保证后续 argmin(abs(stack)) 仍能正确跳过这些哨兵值。
    """
    seg_b, num_classes = args
    H, W = seg_b.shape
    LARGE = float(H + W)            # 比图像最大可能距离更大的哨兵值
    candidates = []
    for c in range(1, num_classes):
        mask_c = (seg_b == c).astype(np.uint8)
        if mask_c.max() == 0:
            # 该类别完全不存在 → 所有像素都"远离该类边界"
            candidates.append(np.full((H, W), LARGE, dtype=np.float32))
            continue
        if mask_c.min() == 1:
            # 该类别填满整图 → 所有像素都"在该类内部最深处"
            candidates.append(np.full((H, W), -LARGE, dtype=np.float32))
            continue
        neg = distance_transform_edt(mask_c).astype(np.float32)
        pos = distance_transform_edt(1 - mask_c).astype(np.float32)
        sdf_c = pos - neg                                      # 原始像素 SDF
        candidates.append(sdf_c.astype(np.float32))            # ← 去归一化
    if not candidates:
        return np.full((H, W), LARGE, dtype=np.float32)
    stack = np.stack(candidates, axis=0)
    best  = np.argmin(np.abs(stack), axis=0)
    return stack[best, np.arange(H)[:, None], np.arange(W)[None, :]]


def compute_sdf_multiclass(label_np: np.ndarray, num_classes: int) -> np.ndarray:
    """
    多类别有符号距离场（F3: 多线程并行版）。
    executor.map() 保序：results[b] 严格对应 label_np[b]。
    """
    B = label_np.shape[0]
    args = [(label_np[b], num_classes) for b in range(B)]
    results = list(_SDF_POOL.map(_sdf_one_image, args))
    return np.stack(results, axis=0)


def compute_soft_boundary_from_sdf(sdf: torch.Tensor,
                                    sigma: float = 3.0) -> torch.Tensor:
    return torch.exp(-sdf.abs() / (sigma + 1e-6))


def compute_boundary_premix(lab_a, lab_b, img_mask, num_classes, sigma=3.0):
    """有标签路径：pre-CutMix GT 边界图（外部调用接口，保持原版签名）"""
    device, B = lab_a.device, lab_a.shape[0]
    sdf_a = torch.from_numpy(
        compute_sdf_multiclass(lab_a.detach().cpu().numpy().astype(np.int32), num_classes)
    ).float().to(device)
    sdf_b = torch.from_numpy(
        compute_sdf_multiclass(lab_b.detach().cpu().numpy().astype(np.int32), num_classes)
    ).float().to(device)
    soft_a = compute_soft_boundary_from_sdf(sdf_a, sigma)
    soft_b = compute_soft_boundary_from_sdf(sdf_b, sigma)
    mf = img_mask.float()
    if mf.dim() == 2:
        mf = mf.unsqueeze(0).expand(B, -1, -1)
    return soft_a * mf + soft_b * (1.0 - mf)


def compute_boundary_premix_unlabeled(prob1_a, prob2_a, prob1_b, prob2_b,
                                       img_mask, num_classes, sigma=3.0,
                                       consistency_threshold=0.75):
    """无标签路径：可靠性过滤伪 SDF 边界图（外部调用接口，保持原版签名）

    BUG C 修复（v8.2）
    ──────────────────
    原版把 `rel < consistency_threshold` 的像素强制置为 0（背景类），
    导致大量"模型不确定"的像素被错误归类为背景，污染 BADFC 中
    背景类的中心向量（_centroids 即使加权也会偏移）。

    修复：改用 `num_classes` 作为 ignore 标签。
      - BADFC._stratified_sample 已有 `(lab < num_classes)` 过滤，自动跳过
      - BADFC._centroids 的 `for c in range(C)` 也不会处理这些像素
      - SDF 计算（compute_sdf_multiclass 只遍历 c=1..C-1）不会把
        ignore 像素当作任何前景类的"内部"，也不会当作背景类的"边界"
        ⇒ 这些像素自然显示为"远离所有边界"，不产生伪边界
    """
    device, B = prob1_a.device, prob1_a.shape[0]

    def _rel(pa, pb):
        pred_a, pred_b = pa.argmax(dim=1), pb.argmax(dim=1)
        rel = (pred_a == pred_b).float() * (pa.max(1)[0] + pb.max(1)[0]) * 0.5
        pseudo = pred_a.clone().long()
        pseudo[rel < consistency_threshold] = num_classes        # ← ignore label
        return pseudo, rel

    with torch.no_grad():
        pseudo_a, rel_a = _rel(prob1_a, prob2_a)
        pseudo_b, rel_b = _rel(prob1_b, prob2_b)

    sdf_a = torch.from_numpy(
        compute_sdf_multiclass(pseudo_a.cpu().numpy().astype(np.int32), num_classes)
    ).float().to(device)
    sdf_b = torch.from_numpy(
        compute_sdf_multiclass(pseudo_b.cpu().numpy().astype(np.int32), num_classes)
    ).float().to(device)
    # v8.4：这里返回“纯 soft boundary”和 reliability 分离结果。
    # 旧版曾在这里提前乘 reliability，forward() 里又乘一次，导致
    # 无标签 DBM 实际为 soft_boundary × reliability^2，边界监督被过度压弱。
    soft_a = compute_soft_boundary_from_sdf(sdf_a, sigma)
    soft_b = compute_soft_boundary_from_sdf(sdf_b, sigma)

    mf = img_mask.float()
    if mf.dim() == 2:
        mf = mf.unsqueeze(0).expand(B, -1, -1)
    return (soft_a * mf + soft_b * (1.0 - mf),
            rel_a  * mf + rel_b  * (1.0 - mf))


# ═══════════════════════════════════════════════════════════════════════════
# 合并核心模块
# ═══════════════════════════════════════════════════════════════════════════

class DecoderBoundaryModule(nn.Module):
    """
    边界感知解码器模块（BADFC + BTNA 合并）—— v8-fixed

    参数变化（相对原版 v8）
    ──────────────────────
    normal_reg_weight : float, 默认 0.05
        F4 新增。BTNA 法向梯度量级激励权重。
        控制 L_normal_reg = -mean(w·log(g_n+1)) 在总 BTNA 损失中的比重。
        设为 0 可完全关闭此项，退回原版行为。
        其余参数与原版完全相同（向后兼容）。
    """

    def __init__(self,
                 num_classes:           int,
                 dec_channels:          int   = 32,
                 proj_dim:              int   = 64,
                 temp_boundary:         float = 0.1,
                 temp_inner:            float = 0.2,    # v8.1: 0.5 → 0.2，避免内部像素 InfoNCE 过热失效
                 border_pixels:         int   = 256,
                 inner_pixels:          int   = 128,
                 aniso_weight:          float = 1.0,
                 cd_aniso_weight:       float = 0.5,
                 cd_entropy_weight:     float = 0.3,
                 normal_reg_weight:     float = 0.05,   # F4 新增
                 start_iter:            int   = 2000,
                 sdf_sigma:             float = 3.0,
                 consistency_threshold: float = 0.75,
                 unlabeled_loss_weight: float = 0.5,
                 eps:                   float = 1e-6):
        super().__init__()
        self.num_classes            = num_classes
        self.temp_boundary          = temp_boundary
        self.temp_inner             = temp_inner
        self.border_pixels          = border_pixels
        self.inner_pixels           = inner_pixels
        self.aniso_weight           = aniso_weight
        self.cd_aniso_weight        = cd_aniso_weight
        self.cd_entropy_weight      = cd_entropy_weight
        self.normal_reg_weight      = normal_reg_weight  # F4
        self.start_iter             = start_iter
        self.sdf_sigma              = sdf_sigma
        self.consistency_threshold  = consistency_threshold
        self.unlabeled_loss_weight  = unlabeled_loss_weight
        self.eps                    = eps

        # ── BADFC 投影头 ─────────────────────────────────────────────────
        ng = min(8, proj_dim)
        self.proj1 = nn.Sequential(
            nn.Conv2d(dec_channels, proj_dim, 1, bias=False),
            nn.GroupNorm(ng, proj_dim), nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim, proj_dim, 1),
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(dec_channels, proj_dim, 1, bias=False),
            nn.GroupNorm(ng, proj_dim), nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim, proj_dim, 1),
        )

        # ── BTNA Sobel 滤波器 ─────────────────────────────────────────────
        sobel_x = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32
        ).unsqueeze(0) / 8.0
        sobel_y = torch.tensor(
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32
        ).unsqueeze(0) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    # ──────────────────────────────────────────────────────────────────────
    # 外部接口：边界图预计算
    # ──────────────────────────────────────────────────────────────────────

    def compute_boundary_premix(self, lab_a, lab_b, img_mask):
        """有标签路径 pre-CutMix 边界图"""
        return compute_boundary_premix(
            lab_a, lab_b, img_mask, self.num_classes, self.sdf_sigma)

    def compute_boundary_premix_unlabeled(self, prob1_a, prob2_a,
                                           prob1_b, prob2_b, img_mask):
        """无标签路径 pre-CutMix 可靠性边界图"""
        return compute_boundary_premix_unlabeled(
            prob1_a, prob2_a, prob1_b, prob2_b, img_mask,
            self.num_classes, self.sdf_sigma, self.consistency_threshold)


    