"""
decoder_boundary_module.py  (v8.4 - dice-aware DBM optimized)
边界感知解码器模块（BADFC + BTNA 合并版）

══════════════════════════════════════════════════════════════════════
v8.2 关键 BUG 修复（相对 v8-fixed）
══════════════════════════════════════════════════════════════════════
  BUG A (严重)  _sdf_one_image: 去掉 SDF 归一化
        原版 `sdf_c / (m + 1e-6)` 把 SDF 归一化到 [-1, 1]。配合 sigma=3，
        在 256x256 ACDC 上所有像素 soft_boundary > 0.71，边界与内部失去
        区分。直接后果：
          - 分层采样退化（is_bnd > 0.5 几乎处处为真）
          - BTNA w_bnd_eff 处处接近 1，边界正则失去聚焦
          - 双温度（temp_boundary vs temp_inner）失效
        修复：去归一化，sigma 保持像素物理意义（默认 3 像素宽边界带）。
        类别完全不存在/完全占满的特殊值改用 ±LARGE 哨兵代替 ±1。

  BUG B (中等)  新增 compute_boundary_from_mixed_pseudo
        原版无标签路径中：DBM forward 收到的 pseudo_labels 是 cross-mixed
        的 mixu_plab，但 soft_boundary 由 compute_boundary_premix_unlabeled
        基于 pred_a（非 cross）计算。两套伪标签每像素都可能不同。
        修复：新方法直接对 cross-mixed 的伪标签算 SDF，让 BADFC 采样、
        类别中心、BTNA 几何约束严格基于同一标签。

  BUG C (中等)  低可靠像素改用 num_classes 作为 ignore label
        原版 `pseudo[rel < threshold] = 0` 把不可靠像素强制归类为背景，
        污染 BADFC 背景类中心向量。
        修复：改用 num_classes 作为越界 ignore 值。
        BADFC._stratified_sample 已有 `(lab < num_classes)` 过滤，
        SDF 计算只遍历 c=1..C-1，自动跳过这些像素。

  BUG D (轻微)  L_normal_reg 加 clamp 上界
        原版 g_n_avg 无上界，训练后期可能 → 大值，
        -log(101) ≈ -4.6 压过其他损失。
        修复：g_n_avg.clamp(max=10) → -log 截在 ≈ -2.4。

══════════════════════════════════════════════════════════════════════
v8-fixed 修复列表（保留）
══════════════════════════════════════════════════════════════════════
  F1  BADFC InfoNCE: valid 类别过滤（已在 v8-fixed 完成）
  F2  L_cd_entropy ent_w 用 H_norm.detach()（已在 v8-fixed 完成）
  F3  SDF 并行化（线程池，已在 v8-fixed 完成）
  F4  BTNA L_normal_reg 防退化（已在 v8-fixed 完成；v8.2 加 clamp）
  F5  分层采样判据用纯 soft_boundary（已在 v8-fixed 完成）
══════════════════════════════════════════════════════════════════════

两个子损失的分工（不变）：
  BADFC（InfoNCE）: 边界附近特征向量 → 正确语义类别中心（语义层）
  BTNA（aniso）:   边界附近特征场导数 → 法向尖锐、切向平滑（几何层）
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Dict

# ═══════════════════════════════════════════════════════════════════════════
# F3: 模块级线程池（import 时创建一次，避免每次迭代的建立开销）
#     scipy.ndimage.distance_transform_edt 为 C 扩展，可释放 GIL。
#     线程加速效果取决于机器 CPU 核心数：
#       2  核（容器）   → 约 1.5-2× speedup
#       8  核（训练机） → 约 4-5× speedup（B=6, 3类=18次EDT）
#       16 核           → 接近线性加速（受 B*C=18 并行数上限约束）
#     如果需要跨进程并行（如某些环境 GIL 实际未释放），可将
#     ThreadPoolExecutor 替换为 ProcessPoolExecutor，代价是 pickle 开销。
# ═══════════════════════════════════════════════════════════════════════════

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

    def compute_boundary_from_mixed_pseudo(self,
                                            mixed_pseudo: torch.Tensor,
                                            mixed_reliability: torch.Tensor
                                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        BUG B 修复（v8.2）：基于 cross-mixed 伪标签直接生成 SDF 边界图

        原版的 compute_boundary_premix_unlabeled 内部用 `pred_a` 计算
        SDF 边界，而训练循环中喂给 DBM forward 的 pseudo_labels 是
        `mixu_plab = plab_sub * mask + plab * (1 - mask)`（cross 伪标签）。
        两套伪标签在每个像素位置都可能不同：
          - BADFC._centroids 用 mixu_plab 算类别中心
          - 但采样位置（边界带）由 db_soft_u 决定，对应不同的伪标签
        ⇒ 语义错位：在"心肌中心"采到的像素，labels 说"心肌"，
                    但 soft_boundary 说"附近不是心肌边界"。

        修复：直接对 cross-mixed 后的 mixu_plab 计算 SDF，
              reliability 也用 cross-mixed 后的 conf_mixu。
              保证 BADFC 采样、_centroids、BTNA 各向异性所用的标签
              和边界图严格基于同一张伪标签 map。

        参数
        ────
        mixed_pseudo      : [B, H, W] long, cross-mixed 伪标签
                            （可包含 num_classes 作为 ignore 值，由 Bug C 修复引入）
        mixed_reliability : [B, H, W] float in [0, 1], cross-mixed 后的可靠性

        返回
        ────
        soft_boundary : [B, H, W] float，纯 SDF soft boundary，不提前乘 reliability
        reliability   : [B, H, W] float，原样返回（forward_unlabeled 里只乘一次）
        """
        device = mixed_pseudo.device
        sdf = torch.from_numpy(
            compute_sdf_multiclass(
                mixed_pseudo.detach().cpu().numpy().astype(np.int32),
                self.num_classes)
        ).float().to(device)
        soft = compute_soft_boundary_from_sdf(sdf, self.sdf_sigma)
        # v8.4：不要在这里乘 reliability；_forward() 里会统一乘一次。
        # 否则无标签边界权重会变成 reliability^2，边界像素梯度过弱。
        return soft, mixed_reliability

    # ──────────────────────────────────────────────────────────────────────
    # 共享内部工具
    # ──────────────────────────────────────────────────────────────────────

    def _resize(self, t: torch.Tensor, H: int, W: int,
                mode: str = 'bilinear') -> torch.Tensor:
        if t.shape[-2] == H and t.shape[-1] == W:
            return t
        return F.interpolate(t.unsqueeze(1).float(), (H, W),
                             mode=mode, align_corners=False).squeeze(1)

    def _resize_labels(self, t: torch.Tensor, H: int, W: int) -> torch.Tensor:
        if t.shape[-2] == H and t.shape[-1] == W:
            return t
        return F.interpolate(t.unsqueeze(1).float(), (H, W),
                             mode='nearest').squeeze(1).long()

    def _sobel2d(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """对任意 [B, C, H, W] 张量做 Sobel，返回 gx/gy [B, C, H, W]"""
        B, C, H, W = x.shape
        xf  = x.reshape(B * C, 1, H, W)
        xp  = F.pad(xf, (1, 1, 1, 1), mode='replicate')
        gx  = F.conv2d(xp, self.sobel_x).reshape(B, C, H, W)
        gy  = F.conv2d(xp, self.sobel_y).reshape(B, C, H, W)
        return gx, gy

    def _boundary_normal(self, scalar_map: torch.Tensor,
                          H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从标量场（soft_boundary 或 entropy_map）提取归一化法向量场 [B,H,W]×2。
        soft_boundary 梯度方向与真实 SDF 法向余弦相似度 ≈ 0.993。
        """
        m   = F.interpolate(scalar_map.unsqueeze(1).float().clamp(min=0.0),
                            (H, W), mode='bilinear', align_corners=False).squeeze(1)
        gx, gy = self._sobel2d(m.unsqueeze(1))
        gx, gy = gx.squeeze(1), gy.squeeze(1)
        mag    = (gx ** 2 + gy ** 2).sqrt().clamp(min=self.eps)
        return gx / mag, gy / mag

    # ──────────────────────────────────────────────────────────────────────
    # BTNA 各向异性比
    # ──────────────────────────────────────────────────────────────────────

    def _anisotropy_ratio(self, feat: torch.Tensor,
                           n_x: torch.Tensor,
                           n_y: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        aniso = g_t / (g_n + g_t + ε) ∈ [0,1]
            → 0 : 法向主导（边界尖锐）
            → 1 : 切向主导（边界模糊）

        Returns: (aniso, g_n, g_t)  均为 [B, H, W]
        """
        gfx, gfy = self._sobel2d(feat)
        t_x, t_y = -n_y, n_x                                      # 切向（法向旋转90°）
        Jn  = gfx * n_x.unsqueeze(1) + gfy * n_y.unsqueeze(1)
        Jt  = gfx * t_x.unsqueeze(1) + gfy * t_y.unsqueeze(1)
        g_n = (Jn ** 2).sum(1)                                     # [B, H, W]
        g_t = (Jt ** 2).sum(1)
        return g_t / (g_n + g_t + self.eps), g_n, g_t

    # ──────────────────────────────────────────────────────────────────────
    # BADFC 类别中心
    # ──────────────────────────────────────────────────────────────────────

    def _centroids(self, proj_flat: torch.Tensor,
                   labels_flat: torch.Tensor, C: int,
                   weights: Optional[torch.Tensor] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        per-batch 类别中心（L2 归一化）。
        返回 (cents [C,D], valid [C] bool)。
        valid[c]=False 的类别 cents[c] 保持为零向量。
        调用方应通过 valid 掩码过滤，避免零中心导致 InfoNCE 梯度污染（F1）。
        """
        D, dev = proj_flat.shape[1], proj_flat.device
        cents  = torch.zeros(C, D, device=dev)
        valid  = torch.zeros(C, dtype=torch.bool, device=dev)
        for c in range(C):
            mask = (labels_flat == c)
            if mask.sum() < 4:
                continue
            if weights is None:
                cents[c] = proj_flat[mask].mean(0)
                valid[c] = True
            else:
                w  = weights[mask]
                ws = w.sum()
                if ws < 1.0:
                    continue
                cents[c] = (proj_flat[mask] * w.unsqueeze(1)).sum(0) / (ws + self.eps)
                valid[c] = True
        return F.normalize(cents, dim=1), valid

    # ──────────────────────────────────────────────────────────────────────
    # BADFC 分层采样
    # ──────────────────────────────────────────────────────────────────────

    def _stratified_sample(self, bnd_split_np: np.ndarray,
                            lab_np: np.ndarray, N: int) -> np.ndarray:
        """
        按边界/内部分层抽取像素索引。

        参数
        ────
        bnd_split_np : 用于边界/内部分类的软边界图（纯 soft_boundary，F5 修复后
                       统一传入 w_bnd 而非 w_bnd_eff，防止无标签路径边界采样不足）
        lab_np       : 展平标签，用于有效像素过滤
        N            : 总像素数上限
        """
        is_bnd = (bnd_split_np > 0.5)
        valid  = (lab_np >= 0) & (lab_np < self.num_classes)
        bi = np.where(valid &  is_bnd)[0]
        ii = np.where(valid & ~is_bnd)[0]
        def _s(a, n):
            return a if len(a) <= n else np.random.choice(a, n, replace=False)
        idx = np.concatenate([_s(bi, self.border_pixels), _s(ii, self.inner_pixels)])
        return idx[idx < N]

    # ──────────────────────────────────────────────────────────────────────
    # 合并 forward（内部核心，有/无标签共用）
    # ──────────────────────────────────────────────────────────────────────

    def _forward(self,
                 dec_feat1:     torch.Tensor,
                 dec_feat2:     torch.Tensor,
                 labels:        torch.Tensor,
                 soft_boundary: torch.Tensor,
                 soft_out1:     Optional[torch.Tensor] = None,
                 soft_out2:     Optional[torch.Tensor] = None,
                 reliability:   Optional[torch.Tensor] = None,
                 ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        统一前向，有/无标签路径共用。

        共享计算（仅执行一次）：
            ① soft_boundary resize → [B, H, W]
            ② Sobel 法向提取 → (n_x, n_y)  供 BTNA 使用

        BADFC（F1 修复）：
            过滤采样像素中对应无效中心的样本，防止零中心向量污染 InfoNCE 梯度。

        BTNA（F4 修复）：
            新增 L_normal_reg 激励法向梯度量级，防止特征场平坦退化解。

        L_cd_entropy（F2 修复）：
            ent_w 使用 H_norm.detach()，切断对 softmax 输出的非预期梯度路径。

        _stratified_sample（F5 修复）：
            分层判据统一改为 w_bnd（纯软边界），不含可靠性缩放。
        """
        B, _, H, W = dec_feat1.shape
        C   = self.num_classes
        dev = dec_feat1.device
        is_unlabeled = (reliability is not None)

        # ── ① 共享：soft_boundary resize + Sobel 法向 ─────────────────────
        w_bnd = self._resize(soft_boundary, H, W).clamp(0.0, 1.0)  # [B,H,W]
        n_x, n_y = self._boundary_normal(soft_boundary, H, W)       # [B,H,W] ×2

        # 可靠性图（无标签）
        if is_unlabeled:
            rel = self._resize(reliability, H, W).clamp(0.0, 1.0)
            w_bnd_eff = w_bnd * rel   # 有效边界权重 = soft_boundary × reliability
        else:
            w_bnd_eff = w_bnd

        # ── BADFC 分支 ────────────────────────────────────────────────────
        # 投影 + L2 归一化
        pf1 = F.normalize(self.proj1(dec_feat1), dim=1)   # [B, proj_dim, H, W]
        pf2 = F.normalize(self.proj2(dec_feat2), dim=1)

        D_p, N = pf1.shape[1], B * H * W
        pf1_flat = pf1.permute(0, 2, 3, 1).reshape(N, D_p)
        pf2_flat = pf2.permute(0, 2, 3, 1).reshape(N, D_p)

        # 对齐标签
        # v8.4：不要 clamp 到 [0, C-1]。
        # 无标签路径会用 num_classes 作为 ignore label；旧版 clamp 会把 ignore
        # 错误变成最后一类（ACDC 中通常是 LV），污染 BADFC 类别中心。
        lab_feat = self._resize_labels(labels, H, W)
        lab_flat = lab_feat.reshape(N)
        bnd_flat = w_bnd_eff.reshape(N)
        rel_flat = rel.reshape(N) if is_unlabeled else None

        # F5: 分层采样使用纯软边界（w_bnd）作为边界/内部判据，
        #     不含可靠性缩放，防止真实边界像素因 rel 低被误分为内部。
        bnd_split_flat = w_bnd.reshape(N)
        idx_np = self._stratified_sample(
            bnd_split_flat.detach().cpu().numpy(),
            lab_flat.detach().cpu().numpy(), N)

        loss_badfc = dec_feat1.sum() * 0.0   # 默认零（保留计算图）
        info_badfc = {'badfc_loss': 0.0, 'badfc_n_pixels': 0,
                      'badfc_boundary_ratio': 0.0}

        if len(idx_np) > 0:
            idx = torch.from_numpy(idx_np).long().to(dev)
            s1  = pf1_flat[idx]
            s2  = pf2_flat[idx]
            sl  = lab_flat[idx]
            sb  = bnd_flat[idx]

            # per-batch 类别中心（交叉：proj1 中心指导 proj2，反之亦然）
            if is_unlabeled:
                cent1, v1 = self._centroids(pf1_flat.detach(), lab_flat, C, rel_flat)
                cent2, v2 = self._centroids(pf2_flat.detach(), lab_flat, C, rel_flat)
            else:
                cent1, v1 = self._centroids(pf1_flat.detach(), lab_flat, C)
                cent2, v2 = self._centroids(pf2_flat.detach(), lab_flat, C)

            if (v1 & v2).any():
                # ── F1: 过滤无效类别像素 ─────────────────────────────────
                # valid[c]=False 的类别中心为零向量，其 dot-product 恒为 0，
                # 对应像素的正样本相似度恒最低，梯度会把特征错误推向其他类别。
                valid_both = v1 & v2                           # [C]
                label_valid = (sl >= 0) & (sl < C)             # ignore label 必须显式过滤
                keep = label_valid & valid_both[sl.clamp(0, C - 1)]  # [M] bool

                if keep.any():
                    s1_k  = s1[keep];   s2_k  = s2[keep]
                    sl_k  = sl[keep];   sb_k  = sb[keep]
                    sw_k  = rel_flat[idx][keep] if is_unlabeled else None

                    # v8.4：InfoNCE denominator 只保留当前 batch 中双分支都有效的类别中心。
                    # 旧版虽然过滤了“样本所属类别无效”的像素，但无效类别的零中心仍在
                    # denominator 中作为伪负类，类别缺失频繁的小 batch 下会造成噪声梯度。
                    valid_ids = torch.where(valid_both)[0]
                    if valid_ids.numel() >= 2:
                        label_map = torch.full((C,), -1, dtype=torch.long, device=dev)
                        label_map[valid_ids] = torch.arange(valid_ids.numel(), device=dev)
                        sl_mapped = label_map[sl_k.clamp(0, C - 1)]
                        mapped_valid = sl_mapped >= 0

                        s1_k = s1_k[mapped_valid]
                        s2_k = s2_k[mapped_valid]
                        sl_k = sl_mapped[mapped_valid]
                        sb_k = sb_k[mapped_valid]
                        if sw_k is not None:
                            sw_k = sw_k[mapped_valid]

                        if s1_k.numel() > 0:
                            cent1_k = cent1[valid_ids]
                            cent2_k = cent2[valid_ids]

                            # 双温度：边界处用低温（锐利决策），内部用高温（平滑分布）
                            temp_k = (sb_k * self.temp_boundary +
                                      (1.0 - sb_k) * self.temp_inner).clamp(min=1e-4)

                            def _infonce(q, k, lq, t, w=None):
                                sim  = torch.mm(q, k.T) / t.unsqueeze(1)
                                sim  = sim - sim.max(1, keepdim=True)[0].detach()
                                exp_ = torch.exp(sim)
                                pos  = exp_[torch.arange(len(lq), device=dev), lq]
                                pl   = -torch.log(pos / (exp_.sum(1) + self.eps) + self.eps)
                                if w is None:
                                    return pl.mean()
                                ws = w.sum()
                                return ((pl * w).sum() / (ws + self.eps)
                                        if ws > self.eps else pl.mean())

                            l1_nc = _infonce(s1_k, cent2_k, sl_k, temp_k, sw_k)
                            l2_nc = _infonce(s2_k, cent1_k, sl_k, temp_k, sw_k)
                            loss_badfc = (l1_nc + l2_nc) * 0.5
                            info_badfc = {
                                'badfc_loss':           loss_badfc.item(),
                                'badfc_boundary_ratio': float(sb_k.mean()),
                                'badfc_n_pixels':       int(s1_k.shape[0]),
                                'badfc_valid_classes':  int(valid_ids.numel()),
                            }

        # ── BTNA 分支 ─────────────────────────────────────────────────────
        # F4: 捕获 g_n 以计算法向梯度量级激励项
        aniso1, g_n1, _ = self._anisotropy_ratio(dec_feat1, n_x, n_y)
        aniso2, g_n2, _ = self._anisotropy_ratio(dec_feat2, n_x, n_y)

        w_sum    = w_bnd_eff.sum().clamp(min=self.eps)
        aniso_avg = (aniso1 + aniso2) * 0.5
        g_n_avg   = (g_n1   + g_n2  ) * 0.5

        # L_aniso：边界附近最小化切向各向异性（g_t/（g_n+g_t）→ 0）
        L_aniso    = (w_bnd_eff * aniso_avg).sum() / w_sum
        # L_cd_aniso：两解码器各向异性模式一致
        L_cd_aniso = (w_bnd_eff * (aniso1 - aniso2).abs()).sum() / w_sum
        # F4 + BUG D 修复（v8.2）L_normal_reg：激励法向梯度量级非零，
        # 防止特征场平坦退化解
        #   -log(g_n+1)：g_n=0 处梯度为 -1（最强推力），有界不爆炸
        #   最小化此项 → g_n 增大 → 边界特征沿法向变化更显著
        # BUG D：原版 g_n_avg 无上界，训练后期 g_n 可能 → 大值，
        #        L_normal_reg = -log(101) ≈ -4.6，负值过大压过其他损失。
        #        加 clamp(max=10) 把 -log 截断在 -log(11) ≈ -2.4，仍保留激励。
        L_normal_reg = -(w_bnd_eff *
                         torch.log(g_n_avg.clamp(max=10.0) + 1.0)).sum() / w_sum

        loss_btna = (self.aniso_weight      * L_aniso
                   + self.cd_aniso_weight   * L_cd_aniso
                   + self.normal_reg_weight * L_normal_reg)
        info_btna = {
            'btna_aniso':      L_aniso.item(),
            'btna_cd_aniso':   L_cd_aniso.item(),
            'btna_normal_reg': L_normal_reg.item(),   # F4 新增日志项
        }

        # L_cd_entropy：无标签专属自监督项（完全不依赖伪标签）
        if is_unlabeled and soft_out1 is not None and soft_out2 is not None:
            so1 = F.interpolate(soft_out1.float(), (H, W),
                                mode='bilinear', align_corners=False)
            so2 = F.interpolate(soft_out2.float(), (H, W),
                                mode='bilinear', align_corners=False)
            p_mean = (so1 + so2) * 0.5
            H_map  = -(p_mean * torch.log(p_mean + self.eps)).sum(1)
            H_norm = (H_map / H_map.detach().amax((1, 2), keepdim=True).clamp(self.eps)
                      ).clamp(0.0, 1.0)

            # 熵图法向（与 SDF 法向独立，保证 L_cd_entropy 无需伪标签）
            en_x, en_y = self._boundary_normal(H_norm.detach(), H, W)
            a1e, _, _  = self._anisotropy_ratio(dec_feat1, en_x, en_y)
            a2e, _, _  = self._anisotropy_ratio(dec_feat2, en_x, en_y)

            # F2: 使用 H_norm.detach() 构造权重，切断梯度路径
            #   原版 H_norm 未 detach → 梯度经 ent_w → H_norm → p_mean → softmax 输出
            #   会产生"几何不一致处降低熵"的非预期隐式信号，与纯几何约束设计意图矛盾。
            ent_w   = H_norm.detach() * rel
            ent_sum = ent_w.sum().clamp(min=self.eps)
            L_cd_entropy = (ent_w * (a1e - a2e).abs()).sum() / ent_sum
            loss_btna = loss_btna + self.cd_entropy_weight * L_cd_entropy
            info_btna['btna_cd_entropy'] = L_cd_entropy.item()

        # ── 合并输出 ──────────────────────────────────────────────────────
        return loss_badfc, loss_btna, {**info_badfc, **info_btna}

    # ──────────────────────────────────────────────────────────────────────
    # 对外接口
    # ──────────────────────────────────────────────────────────────────────

    def forward_labeled(self,
                        dec_feat1:     torch.Tensor,
                        dec_feat2:     torch.Tensor,
                        labels:        torch.Tensor,
                        soft_boundary: torch.Tensor,
                        iter_num:      int
                        ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        有标签路径

        Returns:
            loss_badfc : InfoNCE 语义聚类损失
            loss_btna  : 各向异性几何正则损失（含 F4 L_normal_reg）
            info       : 统计字典（两路合并）
        """
        if iter_num < self.start_iter:
            z = dec_feat1.sum() * 0.0
            return z, z, {'badfc_loss': 0.0, 'btna_aniso': 0.0}
        return self._forward(dec_feat1, dec_feat2, labels, soft_boundary)

    def forward_unlabeled(self,
                          dec_feat1:     torch.Tensor,
                          dec_feat2:     torch.Tensor,
                          pseudo_labels: torch.Tensor,
                          soft_boundary: torch.Tensor,
                          reliability:   torch.Tensor,
                          soft_out1:     torch.Tensor,
                          soft_out2:     torch.Tensor,
                          iter_num:      int
                          ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        无标签路径

        Returns:
            loss_badfc : 可靠性加权 InfoNCE 损失（× unlabeled_loss_weight）
            loss_btna  : 可靠性加权各向异性损失 + L_cd_entropy（F2 修复后纯几何）
            info       : 统计字典（两路合并）
        """
        if iter_num < self.start_iter:
            z = dec_feat1.sum() * 0.0
            return z, z, {'badfc_loss': 0.0, 'btna_aniso_u': 0.0,
                          'btna_cd_entropy': 0.0}

        loss_badfc, loss_btna, info = self._forward(
            dec_feat1, dec_feat2, pseudo_labels, soft_boundary,
            soft_out1=soft_out1, soft_out2=soft_out2, reliability=reliability)

        return (loss_badfc * self.unlabeled_loss_weight,
                loss_btna,
                info)