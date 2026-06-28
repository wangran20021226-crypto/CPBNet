
import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label

from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import val_2d
from utils.CML_utils import generate_mask_2D, features_discrepancy_loss, supervison_loss

from prototype_contrastive_improved import (
    DualModelPrototypeMemory,
    AdaptiveDiscrepancyCompactnessLoss,
    EnhancedPrototypeContrastiveLoss,
    NoiseAugmentedPrototypeLoss,
    ProjectionHead,
    prototype_diversity_regularization,
    get_warmup_weight,
)
from decoder_boundary_module import (
    DecoderBoundaryModule,
    compute_boundary_premix,
    compute_boundary_premix_unlabeled,
)


# ═══════════════════════════════════════════════════════════════════════════
# 参数解析
# ═══════════════════════════════════════════════════════════════════════════

def get_parser():
    p = argparse.ArgumentParser()

    # 基础
    p.add_argument('--root_path',         type=str,   default='../data/ACDC')
    p.add_argument('--exp',               type=str,   default='CML_v8_1')
    p.add_argument('--model',             type=str,   default='unet')
    p.add_argument('--pre_iterations',    type=int,   default=6000)
    p.add_argument('--train_iterations',  type=int,   default=30000)
    p.add_argument('--batch_size',        type=int,   default=24)
    p.add_argument('--labeled_bs',        type=int,   default=12)
    p.add_argument('--labelnum',          type=int,   default=7)
    p.add_argument('--gpu',               type=str,   default='0')
    p.add_argument('--seed',              type=int,   default=1337)
    p.add_argument('--deterministic',     type=int,   default=1)
    p.add_argument('--num_classes',       type=int,   default=4)
    p.add_argument('--patch_size',        type=list,  default=[256, 256])

    # CML 损失权重（沿用 v8 默认）
    p.add_argument('--base_lr',           type=float, default=0.01)
    p.add_argument('--l_weight',          type=float, default=1.0)
    p.add_argument('--u_weight',          type=float, default=0.6)
    p.add_argument('--dis_weight',        type=float, default=0.1)
    p.add_argument('--mask_ratio',        type=float, default=0.4)
    p.add_argument('--lr_power',          type=float, default=0.9)
    p.add_argument('--rampup_length',     type=float, default=0.3)

    # 置信度（沿用 v8 默认）
    p.add_argument('--temperature',        type=float, default=0.5)
    p.add_argument('--conf_thresh',        type=float, default=0.72)
    p.add_argument('--conf_weight',        type=float, default=1.0)
    p.add_argument('--use_soft_threshold', type=int,   default=1)

    # CPAM 原型学习（沿用 v8 默认；用户已通过 shell 调到 baseline 最佳点）
    p.add_argument('--use_prototype',         type=int,   default=1)
    p.add_argument('--proto_dim',             type=int,   default=128)
    p.add_argument('--proto_weight',          type=float, default=0.6)
    p.add_argument('--proto_compact_weight',  type=float, default=0.25)
    p.add_argument('--proto_div_weight',      type=float, default=0.08)
    p.add_argument('--proto_momentum',        type=float, default=0.99)
    p.add_argument('--proto_temp',            type=float, default=0.07)
    p.add_argument('--proto_warmup',          type=int,   default=2000)
    p.add_argument('--use_noise_contrast',    type=int,   default=0)
    p.add_argument('--noise_std',             type=float, default=0.1)
    # C2: 拆出 CPAM 自己的一致性阈值（原型更新用）；默认沿用 v8 共用值 0.75
    p.add_argument('--cpam_consistency_threshold', type=float, default=0.75,
                   help='CPAM 原型 EMA 更新的"两分支预测一致 × 平均置信度"门限')

    # DecoderBoundaryModule（BADFC + BTNA 合并）
    p.add_argument('--use_decoder_boundary',      type=int,   default=1,
                   help='启用 DecoderBoundaryModule（BADFC+BTNA 合并）')
    p.add_argument('--db_start_iter',             type=int,   default=5000,   # C4: 2000 → 5000
                   help='DBM 启用迭代起点；之前等价于完全不参与训练')
    p.add_argument('--db_warmup_iters',           type=int,   default=2000,   # C3 新增
                   help='DBM 损失从 0 线性渐入到满权重所用的 iter 数；'
                        '0 表示沿用 v8 硬开关行为')
    p.add_argument('--db_proj_dim',               type=int,   default=64)
    p.add_argument('--db_temp_boundary',          type=float, default=0.1)
    p.add_argument('--db_temp_inner',             type=float, default=0.2)    # C5: 0.5 → 0.2
    p.add_argument('--db_aniso_weight',           type=float, default=1.0)
    p.add_argument('--db_cd_aniso_weight',        type=float, default=0.5)
    p.add_argument('--db_cd_entropy_weight',      type=float, default=0.3)
    p.add_argument('--db_badfc_labeled_weight',   type=float, default=0.3,
                   help='BADFC 有标签损失权重')
    p.add_argument('--db_badfc_unlabeled_weight', type=float, default=0.15,
                   help='BADFC 无标签损失权重')
    p.add_argument('--db_btna_labeled_weight',    type=float, default=0.15,
                   help='BTNA 有标签损失权重')
    p.add_argument('--db_btna_unlabeled_weight',  type=float, default=0.08,
                   help='BTNA 无标签损失权重')
    # C2: DBM 内部 SDF 伪标签门限，独立于 CPAM；比 CPAM 松一些
    p.add_argument('--db_consistency_threshold',  type=float, default=0.6,
                   help='DBM 无标签路径中 SDF 伪标签的可靠性门限；'
                        '比 CPAM 的更松，避免前景像素大量被归零导致 SDF 边界消失')

    # v8.4：Dice-aware DBM 输出级监督。
    # 旧 DBM 只约束 decoder feature，主要改善 HD95；下面四项让 DBM soft-boundary
    # 直接参与 out1/out2 的 CE/Dice 优化，因此更容易带来 mean Dice 提升。
    p.add_argument('--db_use_output_loss',        type=int,   default=1,
                   help='启用 DBM 边界加权 CE/Dice，直接监督输出 logits')
    p.add_argument('--db_boundary_alpha',         type=float, default=2.0,
                   help='边界区域加权强度：weight = 1 + alpha * soft_boundary')
    p.add_argument('--db_boundary_ce_l_weight',   type=float, default=0.10)
    p.add_argument('--db_boundary_dice_l_weight', type=float, default=0.20)
    p.add_argument('--db_boundary_ce_u_weight',   type=float, default=0.05)
    p.add_argument('--db_boundary_dice_u_weight', type=float, default=0.10)

    # ── 优化 1：class-adaptive alpha（v8.5 新增）────────────────────────
    # ACDC 四类的边界加权系数：BG / RV / Myo / LV
    #   BG  (class 0): 0.5  — 背景边界噪声较多，轻度加权避免干扰
    #   RV  (class 1): 2.0  — 中等结构，与原版全局 alpha 持平
    #   Myo (class 2): 4.0  — 薄环形，整体几乎在边界带内，需强约束
    #   LV  (class 3): 1.5  — 大结构，内部已有足够监督，边界适度加权
    # 若 db_use_class_alpha=0，退回全局 db_boundary_alpha（向后兼容）。
    p.add_argument('--db_use_class_alpha',  type=int,   default=1,
                   help='启用 class-adaptive alpha（0=退回全局标量 alpha）')
    p.add_argument('--db_alpha_bg',         type=float, default=0.5,
                   help='背景类边界加权系数（ACDC class 0）')
    p.add_argument('--db_alpha_rv',         type=float, default=2.0,
                   help='RV 边界加权系数（ACDC class 1）')
    p.add_argument('--db_alpha_myo',        type=float, default=4.0,
                   help='Myo 边界加权系数（ACDC class 2）；心肌薄，需更强约束')
    p.add_argument('--db_alpha_lv',         type=float, default=1.5,
                   help='LV 边界加权系数（ACDC class 3）')

    # ── 优化 2：无标签 DBM-Dice 独立阈值（v8.5 新增）───────────────────
    # DBM output-level Dice 只在 soft_boundary 高处有效，对噪声标签更鲁棒，
    # 不需要和 CE 损失共用同一严格阈值（conf_thresh=0.72）。
    # 降低到 0.45 可让训练前/中期有更多无标签像素贡献 DBM-Dice 梯度。
    p.add_argument('--db_dice_conf_thresh', type=float, default=0.45,
                   help='无标签 DBM output-level Dice 的置信度阈值；'
                        '独立于 conf_thresh，允许更多无标签像素贡献梯度')

    return p


# ═══════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def polynomial_lr_decay(base_lr, iter_num, max_iter, power=0.9):
    return base_lr * (1.0 - float(iter_num) / max(float(max_iter), 1.0)) ** power


def get_rampup_weight(current_iter, max_iter, rampup_length_ratio=0.33):
    rampup = int(max_iter * rampup_length_ratio)
    if current_iter < rampup:
        phase = 1.0 - current_iter / rampup
        return float(np.exp(-5.0 * phase * phase))
    return 1.0


def get_db_ramp(iter_num, db_start_iter, db_warmup_iters):
    """
    C3: DBM 损失渐入权重（线性 0→1）

    iter_num <  db_start_iter                   → 0
    db_start_iter ≤ iter_num < +warmup          → 线性 0..1
    iter_num ≥ db_start_iter + db_warmup_iters  → 1

    db_warmup_iters <= 0 时退回 v8 硬开关行为（即一次性满权重接入）。
    """
    if iter_num < db_start_iter:
        return 0.0
    if db_warmup_iters is None or db_warmup_iters <= 0:
        return 1.0
    progress = (iter_num - db_start_iter) / float(db_warmup_iters)
    return float(min(max(progress, 0.0), 1.0))


def sharpening(P, temperature=0.5):
    T = 1.0 / max(temperature, 0.01)
    P_s = P ** T
    return P_s / (P_s.sum(dim=1, keepdim=True) + 1e-10)


def adaptive_confidence_weighting(conf_map, threshold=0.75,
                                   temperature=0.5, use_soft=True):
    conf_s = conf_map ** (1.0 / max(temperature, 0.01))
    return torch.sigmoid((conf_s - threshold) * 10.0) if use_soft \
           else (conf_s > threshold).float()


def compute_confidence_map(probs1, probs2):
    conf1, pred1 = probs1.max(dim=1)
    conf2, pred2 = probs2.max(dim=1)
    return (conf1 + conf2) * 0.5 * (pred1 == pred2).float()


def _resize_spatial_map(t, H, W, mode='bilinear'):
    """Resize [B,H,W] or [B,1,H,W] map to [B,H,W]."""
    if t is None:
        return None
    if t.dim() == 4 and t.shape[1] == 1:
        t = t[:, 0]
    if t.shape[-2:] == (H, W):
        return t.float()
    return F.interpolate(t.unsqueeze(1).float(), size=(H, W),
                         mode=mode, align_corners=False).squeeze(1)


# ── 优化 1 辅助函数（v8.5）────────────────────────────────────────────────

def build_alpha_map(target: torch.Tensor,
                    class_alphas: list,
                    num_classes: int,
                    ignore_index: int = None) -> torch.Tensor:
    """
    逐像素构造 class-adaptive alpha map [B, H, W]。

    参数
    ────
    target       : [B, H, W] long，像素类别标签（可含 ignore_index）
    class_alphas : list[float]，长度 == num_classes，
                   class_alphas[c] 为第 c 类的边界加权系数
    ignore_index : ignore 像素的 alpha 强制置 0（不参与边界加权）

    设计说明
    ────────
    原版 boundary_weighted_ce_loss 里：weight = 1 + alpha * bnd
    本函数改为按类别查表：weight = 1 + alpha_map * bnd
    其中 alpha_map[b,h,w] = class_alphas[ target[b,h,w] ]
    alpha_map 对 ignore 像素置 0，让 weight=1（退化为普通 CE 权重），
    但配合外层 ignore_index 过滤后这些像素不会参与最终 loss 归一化。
    """
    alphas = torch.tensor(class_alphas, dtype=torch.float32,
                          device=target.device)
    safe = target.clone()
    if ignore_index is not None:
        safe[safe == ignore_index] = 0          # 临时映射到 class 0，下面再置 0
    safe = safe.clamp(0, num_classes - 1)
    alpha_map = alphas[safe]                    # [B, H, W]
    if ignore_index is not None:
        alpha_map = alpha_map.clone()
        alpha_map[target == ignore_index] = 0.0
    return alpha_map                            # float [B, H, W]，no_grad 外部 detach


def build_boundary_spatial_weight(soft_boundary: torch.Tensor,
                                   target: torch.Tensor,
                                   class_alphas: list,
                                   num_classes: int,
                                   ignore_index: int = None) -> torch.Tensor:
    """
    构造 class-adaptive 空间权重，用于 masked_soft_dice_loss 的 spatial_weight。

    weight[b,h,w] = 1 + alpha_map[b,h,w] * soft_boundary[b,h,w]

    与 boundary_weighted_ce_loss 内部的标量 alpha 版本等价，但 alpha 按类别差异化。
    返回值已 .detach()，不参与计算图。
    """
    bnd  = soft_boundary.clamp(0.0, 1.0)
    amap = build_alpha_map(target, class_alphas, num_classes, ignore_index)
    return (1.0 + amap * bnd).detach()         # [B, H, W]


def boundary_weighted_ce_loss(logits, target, soft_boundary,
                              reliability=None, ignore_index=None,
                              alpha=2.0, alpha_map=None, eps=1e-6):
    """
    DBM 输出级边界 CE：让 soft_boundary 直接改变 out1/out2 的 CE 梯度。

    logits        : [B,C,H,W]
    target        : [B,H,W]
    soft_boundary : [B,H,W]，纯边界图，不建议预乘 reliability
    reliability   : [B,H,W] or None，无标签可靠性权重
    alpha         : 全局标量加权强度（当 alpha_map=None 时使用）
    alpha_map     : [B,H,W] float or None，逐像素 class-adaptive 加权强度
                    （由 build_alpha_map 生成；不为 None 时忽略 alpha 标量）

    v8.5 修改：新增 alpha_map 参数，支持 class-adaptive 边界加权。
    向后兼容：alpha_map=None 时行为与 v8.4 完全一致。
    """
    B, C, H, W = logits.shape
    target = target.long()
    bnd = _resize_spatial_map(soft_boundary, H, W).clamp(0.0, 1.0)

    if reliability is not None:
        rel = _resize_spatial_map(reliability, H, W).clamp(0.0, 1.0)
    else:
        rel = None

    if ignore_index is None:
        ce = F.cross_entropy(logits, target, reduction='none')
        valid = torch.ones_like(target, dtype=torch.float32)
    else:
        ce = F.cross_entropy(logits, target, reduction='none', ignore_index=ignore_index)
        valid = (target != ignore_index).float()

    # ── class-adaptive or scalar alpha ────────────────────────────────
    # alpha_map 传入时逐像素查表；否则退回原版标量行为。
    if alpha_map is not None:
        am = _resize_spatial_map(alpha_map, H, W)
        weight = (1.0 + am * bnd) * valid
    else:
        weight = (1.0 + alpha * bnd) * valid

    if rel is not None:
        weight = weight * rel

    return (ce * weight).sum() / weight.sum().clamp(min=eps)


def masked_soft_dice_loss(logits, target, spatial_weight,
                          num_classes, reliability=None,
                          ignore_index=None, include_background=False,
                          smooth=1e-5):
    """
    DBM 输出级边界 Dice：用 DBM soft-boundary 构造空间权重，直接优化区域重叠。

    与普通 Dice 不同：这里可以只在边界/难分区域加权，也能忽略低置信伪标签。
    """
    B, C, H, W = logits.shape
    target = target.long()
    probs = torch.softmax(logits, dim=1)

    w = _resize_spatial_map(spatial_weight, H, W).float()
    if reliability is not None:
        rel = _resize_spatial_map(reliability, H, W).clamp(0.0, 1.0)
        w = w * rel

    if ignore_index is not None:
        valid = (target != ignore_index).float()
        target_safe = target.clone()
        target_safe[target_safe == ignore_index] = 0
    else:
        valid = torch.ones_like(target, dtype=torch.float32)
        target_safe = target

    target_onehot = F.one_hot(
        target_safe.clamp(0, num_classes - 1), num_classes=num_classes
    ).permute(0, 3, 1, 2).float()

    w = (w * valid).unsqueeze(1)
    class_ids = range(num_classes) if include_background else range(1, num_classes)

    losses = []
    for c in class_ids:
        p = probs[:, c:c + 1]
        y = target_onehot[:, c:c + 1]
        inter = (p * y * w).sum()
        denom = ((p + y) * w).sum()
        dice = (2.0 * inter + smooth) / (denom + smooth)
        losses.append(1.0 - dice)

    if len(losses) == 0:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    for i in range(segmentation.shape[0]):
        class_list = []
        for c in range(1, 4):
            tmp = (segmentation[i] == c).float().detach().cpu().numpy()
            try:
                lbl = label(tmp)
                if lbl.max() != 0:
                    tmp = (lbl == (np.argmax(np.bincount(lbl.flat)[1:]) + 1)) * c
                class_list.append(tmp)
            except Exception:
                class_list.append(tmp)
        batch_list.append(class_list[0] + class_list[1] + class_list[2])
    return torch.Tensor(batch_list).cuda()


def patients_to_slices(dataset, patiens_num):
    if "ACDC" in dataset:
        ref = {"1": 32, "3": 68, "7": 136, "14": 256,
               "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate" in dataset:
        ref = {"2": 27, "4": 53, "8": 120, "12": 179,
               "16": 256, "21": 312, "42": 623}
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return ref[str(patiens_num)]


# ═══════════════════════════════════════════════════════════════════════════
# 检查点工具
# ═══════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, filepath,
                    proj_head1=None, proj_head2=None, db_module=None):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    if proj_head1 is not None: state['proj_head1'] = proj_head1.state_dict()
    if proj_head2 is not None: state['proj_head2'] = proj_head2.state_dict()
    if db_module  is not None: state['db_module']  = db_module.state_dict()
    torch.save(state, filepath)


def load_checkpoint(model, filepath, optimizer=None,
                    proj_head1=None, proj_head2=None, db_module=None,
                    load_optimizer=True):
    ckpt = torch.load(filepath)
    model.load_state_dict(ckpt['model'])
    if load_optimizer and optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if proj_head1 and 'proj_head1' in ckpt: proj_head1.load_state_dict(ckpt['proj_head1'])
    if proj_head2 and 'proj_head2' in ckpt: proj_head2.load_state_dict(ckpt['proj_head2'])
    if db_module  and 'db_module'  in ckpt: db_module.load_state_dict(ckpt['db_module'])
    logging.info(f"Loaded: {filepath}")


# ═══════════════════════════════════════════════════════════════════════════
# 预训练
# ═══════════════════════════════════════════════════════════════════════════

def pre_train(args, snapshot_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = net_factory(in_chns=1, class_num=args.num_classes).cuda()

    def worker_init_fn(wid): random.seed(args.seed + wid)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None,
                             transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val   = BaseDataSets(base_dir=args.root_path, split="val")

    labeled_slice  = patients_to_slices(args.root_path, args.labelnum)
    labeled_idxs   = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, len(db_train)))
    batch_sampler  = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                              num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader   = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    writer    = SummaryWriter(snapshot_path + '/log')

    model.train()
    iter_num, best_performance = 0, 0.0
    labeled_sub_bs = args.labeled_bs // 2
    max_epoch = args.pre_iterations // len(trainloader) + 1

    for _ in tqdm(range(max_epoch), ncols=80, desc="Pre-train"):
        for batch in trainloader:
            vol  = batch['image'].cuda()
            lab  = batch['label'].cuda()
            img_a = vol[:labeled_sub_bs]
            img_b = vol[labeled_sub_bs:args.labeled_bs]
            lab_a = lab[:labeled_sub_bs]
            lab_b = lab[labeled_sub_bs:args.labeled_bs]

            with torch.no_grad():
                img_mask, _ = generate_mask_2D(img_a, args.mask_ratio)
            mix_img = img_a * img_mask + img_b * (1 - img_mask)
            mix_lab = lab_a * img_mask + lab_b * (1 - img_mask)

            out1, out2, feat1, feat2, _, _ = model(mix_img)
            loss = (args.l_weight * (supervison_loss(out1, mix_lab, class_num=args.num_classes) +
                                     supervison_loss(out2, mix_lab, class_num=args.num_classes)) +
                    args.dis_weight * (features_discrepancy_loss(feat1, feat2) +
                                       features_discrepancy_loss(feat2, feat1)))

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            iter_num += 1
            lr_ = polynomial_lr_decay(args.base_lr, iter_num, args.pre_iterations, args.lr_power)
            for pg in optimizer.param_groups: pg['lr'] = lr_
            writer.add_scalar('pre/loss', loss.item(), iter_num)

            if iter_num % 50 == 0:
                logging.info(f'[Pre] {iter_num}/{args.pre_iterations} loss={loss.item():.4f}')

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                ml = 0.0
                with torch.no_grad():
                    for vb in valloader:
                        ml += np.array(val_2d.test_single_volume(
                            vb["image"], vb["label"], model, classes=args.num_classes))
                ml /= len(db_val)
                perf = np.mean(ml, axis=0)[0]
                writer.add_scalar('pre/val_dice', perf, iter_num)
                if perf > best_performance:
                    best_performance = perf
                    save_checkpoint(model, optimizer,
                                    os.path.join(snapshot_path, f'{args.model}_best_model.pth'))
                logging.info(f'[Pre-Val] dice={perf:.4f}  best={best_performance:.4f}')
                model.train()

            if iter_num >= args.pre_iterations: break
        if iter_num >= args.pre_iterations: break

    writer.close()
    logging.info(f"Pre-train done. Best dice={best_performance:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 主训练
# ═══════════════════════════════════════════════════════════════════════════

def train(args, pre_snapshot_path, snapshot_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    num_classes = args.num_classes

    model = net_factory(in_chns=1, class_num=num_classes).cuda()

    # ── 自动探测 dec_feat 的通道数与分辨率（C6） ─────────────────────────
    # 不同版本的 UNet patch 可能让 dec_feat 输出在 up3 (32@128×128) 或
    # up4 (16@256×256)。这里用一次 dummy forward 取真实形状，避免硬编码不匹配。
    with torch.no_grad():
        _dummy = torch.zeros(1, 1, args.patch_size[0], args.patch_size[1]).cuda()
        _out = model(_dummy)
        if not (isinstance(_out, (list, tuple)) and len(_out) >= 6):
            raise RuntimeError(
                f"UNet.forward 返回元组长度 {len(_out) if isinstance(_out, (list, tuple)) else 'N/A'}，"
                f"DBM 需要 (out1, out2, emb1, emb2, dec_feat1, dec_feat2) 共 6 项")
        _df1 = _out[4]
        detected_dec_channels = int(_df1.shape[1])
        detected_dec_size     = int(_df1.shape[-1])
    logging.info(
        f"Detected dec_feat shape: [B, {detected_dec_channels}, "
        f"{detected_dec_size}, {detected_dec_size}]")
    if detected_dec_channels not in (16, 32):
        logging.warning(
            f"dec_feat 通道数 {detected_dec_channels} 非典型值（期望 16 或 32），"
            f"DBM 仍会按此值构造，但请确认 UNet 输出无误")
    if detected_dec_size >= 256:
        logging.warning(
            f"dec_feat 分辨率 {detected_dec_size}×{detected_dec_size} 较高，"
            f"DBM 的 SDF/Sobel/InfoNCE 开销将比 128×128 增大约 4×；如需提速，"
            f"建议把 UNet 的 dec_feat 改回 up3 层（32 通道, 128×128）")
    del _dummy, _out, _df1
    torch.cuda.empty_cache()

    # ── CPAM ─────────────────────────────────────────────────────────────
    proj_head1 = proj_head2 = proto_memory = None
    loss_proto_contrast = loss_discrepancy_compact = None

    if args.use_prototype:
        proj_head1 = ProjectionHead(256, args.proto_dim, args.proto_dim, dropout=0.05).cuda()
        proj_head2 = ProjectionHead(256, args.proto_dim, args.proto_dim, dropout=0.05).cuda()
        proto_memory = DualModelPrototypeMemory(
            num_classes, args.proto_dim, args.proto_momentum, device='cuda')
        loss_proto_contrast = (
            NoiseAugmentedPrototypeLoss(num_classes, args.proto_temp, args.noise_std)
            if args.use_noise_contrast else
            EnhancedPrototypeContrastiveLoss(num_classes, args.proto_temp))
        loss_discrepancy_compact = AdaptiveDiscrepancyCompactnessLoss(num_classes, args.proto_dim)

    # ── DecoderBoundaryModule（BADFC + BTNA 合并）───────────────────────
    db_module = None
    if args.use_decoder_boundary:
        db_module = DecoderBoundaryModule(
            num_classes=num_classes,
            dec_channels=detected_dec_channels,           # C6: 用真实通道数
            proj_dim=args.db_proj_dim,
            temp_boundary=args.db_temp_boundary,
            temp_inner=args.db_temp_inner,
            aniso_weight=args.db_aniso_weight,
            cd_aniso_weight=args.db_cd_aniso_weight,
            cd_entropy_weight=args.db_cd_entropy_weight,
            start_iter=args.db_start_iter,
            sdf_sigma=3.0,
            consistency_threshold=args.db_consistency_threshold,  # C2: 用 DBM 自己的阈值
            unlabeled_loss_weight=0.5,
        ).cuda()
        logging.info(
            f"DecoderBoundaryModule (BADFC+BTNA): enabled  "
            f"dec_channels={detected_dec_channels}  "
            f"start_iter={args.db_start_iter}  warmup={args.db_warmup_iters}  "
            f"db_consistency_threshold={args.db_consistency_threshold}  "
            f"temp_inner={args.db_temp_inner}")

    # ── 优化 1：class-adaptive alpha 列表（v8.5）─────────────────────────
    # 长度必须等于 num_classes；当 db_use_class_alpha=0 时退回全局标量。
    # 当前对 ACDC 的默认分配：BG=0.5 / RV=2.0 / Myo=4.0 / LV=1.5
    # 若 num_classes != 4（如 LA 数据集），自动回退到全局 alpha，不报错。
    if args.use_decoder_boundary and args.db_use_output_loss:
        if args.db_use_class_alpha and num_classes == 4:
            _class_alphas = [
                args.db_alpha_bg,   # class 0: background
                args.db_alpha_rv,   # class 1: RV
                args.db_alpha_myo,  # class 2: Myo
                args.db_alpha_lv,   # class 3: LV
            ]
            logging.info(
                f"DBM class-adaptive alpha: BG={_class_alphas[0]:.1f}  "
                f"RV={_class_alphas[1]:.1f}  Myo={_class_alphas[2]:.1f}  "
                f"LV={_class_alphas[3]:.1f}  (db_dice_conf_thresh={args.db_dice_conf_thresh})")
        else:
            _class_alphas = None   # 退回全局标量 alpha
            if args.db_use_class_alpha and num_classes != 4:
                logging.warning(
                    f"db_use_class_alpha=1 但 num_classes={num_classes} ≠ 4，"
                    f"退回全局 db_boundary_alpha={args.db_boundary_alpha}")
    else:
        _class_alphas = None

    # ── 数据 ─────────────────────────────────────────────────────────────
    def worker_init_fn(wid): random.seed(args.seed + wid)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None,
                             transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val   = BaseDataSets(base_dir=args.root_path, split="val")

    labeled_slice  = patients_to_slices(args.root_path, args.labelnum)
    labeled_idxs   = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, len(db_train)))
    batch_sampler  = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                              num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader   = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    load_checkpoint(model, os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth'),
                    load_optimizer=False)

    # ── 优化器 ────────────────────────────────────────────────────────────
    params = list(model.parameters())
    if args.use_prototype:
        params += list(proj_head1.parameters()) + list(proj_head2.parameters())
    if args.use_decoder_boundary and db_module:
        params += list(db_module.parameters())

    optimizer = optim.SGD(params, lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    writer    = SummaryWriter(snapshot_path + '/log')

    logging.info("=" * 60)
    logging.info("Main training  CPAM + DecoderBoundaryModule(BADFC+BTNA)")
    logging.info("=" * 60)

    model.train()
    if args.use_prototype: proj_head1.train(); proj_head2.train()
    if args.use_decoder_boundary and db_module: db_module.train()

    iter_num, best_performance = 0, 0.0
    labeled_sub_bs   = args.labeled_bs // 2
    unlabeled_sub_bs = (args.batch_size - args.labeled_bs) // 2
    max_epoch = args.train_iterations // len(trainloader) + 1

    for _ in tqdm(range(max_epoch), ncols=80, desc="Train"):
        for batch in trainloader:
            vol  = batch['image'].cuda()
            lab  = batch['label'].cuda()

            img_a   = vol[:labeled_sub_bs]
            img_b   = vol[labeled_sub_bs:args.labeled_bs]
            lab_a   = lab[:labeled_sub_bs]
            lab_b   = lab[labeled_sub_bs:args.labeled_bs]
            unimg_a = vol[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            unimg_b = vol[args.labeled_bs + unlabeled_sub_bs:]

            # ── 有标签 CutMix 前 ─────────────────────────────────────────
            with torch.no_grad():
                img_mask, _ = generate_mask_2D(img_a, args.mask_ratio)
                soft_l = None
                if args.use_decoder_boundary and db_module:
                    soft_l = db_module.compute_boundary_premix(lab_a, lab_b, img_mask)

            mixl_img = img_a * img_mask + img_b * (1 - img_mask)
            mixl_lab = (lab_a * img_mask + lab_b * (1 - img_mask)).long()

            out1_l, out2_l, feat1_l, feat2_l, df1_l, df2_l = model(mixl_img)

            loss_l = args.l_weight * (
                supervison_loss(out1_l, mixl_lab, class_num=num_classes) +
                supervison_loss(out2_l, mixl_lab, class_num=num_classes))
            loss_dis_l = (features_discrepancy_loss(feat1_l, feat2_l) +
                          features_discrepancy_loss(feat2_l, feat1_l))

            # ── 无标签 CutMix 前 ─────────────────────────────────────────
            db_soft_u = db_rel_u = None
            mixu_plab_for_sdf = None
            with torch.no_grad():
                unout1, unout2, _, _, _, _ = model(vol[args.labeled_bs:])
                prob1s = sharpening(F.softmax(unout1, dim=1), args.temperature)
                prob2s = sharpening(F.softmax(unout2, dim=1), args.temperature)
                conf_map = compute_confidence_map(prob1s, prob2s)
                avg_conf = conf_map.mean().item()

                plab     = get_ACDC_2DLargestCC(prob1s.argmax(dim=1))
                plab_sub = get_ACDC_2DLargestCC(prob2s.argmax(dim=1))

                unimg_mask, _ = generate_mask_2D(unimg_a, args.mask_ratio)

                # BUG B 修复（v8.2）：DBM 的 soft_boundary 改为基于 cross-mixed
                # 后的 mixu_plab 计算，与 BADFC 内部用的 pseudo_labels 严格一致。
                # 原版用 compute_boundary_premix_unlabeled 基于 pred_a（非 cross）
                # 算 SDF，与 mixu_plab（cross 伪标签）每像素都可能不同，造成
                # 采样位置和类别中心的语义错位。
                # 注：这里把 DBM 边界图的计算挪到 mixu_plab 生成之后。

            mixu_img  = unimg_a * unimg_mask + unimg_b * (1 - unimg_mask)
            mixu_plab = (plab_sub[:unlabeled_sub_bs] * unimg_mask +
                         plab[unlabeled_sub_bs:] * (1 - unimg_mask)).long()
            mixu_plab_sub = (plab[:unlabeled_sub_bs] * unimg_mask +
                             plab_sub[unlabeled_sub_bs:] * (1 - unimg_mask)).long()

            with torch.no_grad():
                conf_mixu = (conf_map[:unlabeled_sub_bs] * unimg_mask +
                             conf_map[unlabeled_sub_bs:] * (1 - unimg_mask))

                # BUG B + BUG C 修复：基于 cross-mixed 的 mixu_plab 算 SDF 边界图，
                # 同时把低可靠性像素标记为 ignore（num_classes），避免污染
                # BADFC 背景类中心向量。
                if args.use_decoder_boundary and db_module:
                    # 应用一致性 + 置信度联合过滤，与 compute_boundary_premix_unlabeled
                    # 内部行为对齐：低可靠性 → ignore（num_classes），
                    # 但所用伪标签是 cross 后的 mixu_plab，保证语义统一。
                    # v8.4：SDF/soft-boundary 用 raw cross-mixed pseudo 计算，
                    # 避免 ignore 洞把连续结构切断、制造内部假边界；
                    # loss label 再单独过滤为 ignore。
                    mixu_plab_for_sdf = mixu_plab.clone()
                    mixu_plab_for_sdf[conf_mixu < args.db_consistency_threshold] = num_classes
                    db_soft_u, db_rel_u = db_module.compute_boundary_from_mixed_pseudo(
                        mixu_plab, conf_mixu)

            out1_u, out2_u, feat1_u, feat2_u, df1_u, df2_u = model(mixu_img)

            # ── BUG 1 修复（v8.3）────────────────────────────────────────────
            # 原版公式: ce * (1.0 + conf_weight * adap_w), 范围 [1.0, 2.0]
            # 低置信 / 模型不一致的像素（adap_w≈0）仍有基础权重 1.0，
            # 其错误伪标签梯度始终参与反传，是 Dice 不涨的根本原因。
            #
            # 修复：改为乘法权重，范围 [0, 1]：
            #   adap_w ≈ 0  → CE 权重 ≈ 0，噪声像素被真正屏蔽
            #   adap_w ≈ 1  → CE 权重 ≈ 1，高置信像素充分参与
            # conf_weight 作为高置信像素的额外增益保留（使范围达到 [0, 1+conf_weight]）
            # ─────────────────────────────────────────────────────────────────
            adap_w = adaptive_confidence_weighting(
                conf_mixu, args.conf_thresh, args.temperature, bool(args.use_soft_threshold))

            ce1 = F.cross_entropy(out1_u, mixu_plab,     reduction='none')
            ce2 = F.cross_entropy(out2_u, mixu_plab_sub, reduction='none')

            iter_num += 1
            ramp = get_rampup_weight(iter_num, args.train_iterations, args.rampup_length)

            # 乘法权重：adap_w * (1 + conf_weight)，低置信像素 → 0，高置信 → 1+conf_weight
            effective_w = adap_w * (1.0 + args.conf_weight)
            valid_pixels = effective_w.sum().clamp(min=1.0)
            loss_u = args.u_weight * ramp * (
                (ce1 * effective_w).sum() / valid_pixels +
                (ce2 * effective_w).sum() / valid_pixels)

            loss_dis_u = (features_discrepancy_loss(feat1_u, feat2_u) +
                          features_discrepancy_loss(feat2_u, feat1_u))
            loss_dis = args.dis_weight * (loss_dis_l + loss_dis_u)

            # ── BUG 2 修复（v8.3）：CPAM 使用置信度过滤后的伪标签 ────────────
            # 原版 CPAM 的 loss_proto_contrast 和 loss_discrepancy_compact 都
            # 直接使用 mixu_plab（未过滤），低置信错误标签污染原型，
            # 把无标签特征拉向错误方向 → 原型被反向优化 → Dice 停滞。
            #
            # 修复：用 conf_mixu >= cpam_consistency_threshold 作为过滤门限，
            # 不满足的像素标记为 num_classes（ignore），CPAM 内部已有
            # `(labels < C)` 的过滤逻辑，自动跳过。
            # ─────────────────────────────────────────────────────────────────
            with torch.no_grad():
                mixu_plab_filtered = mixu_plab.clone()
                mixu_plab_filtered[conf_mixu < args.cpam_consistency_threshold] = num_classes
                # out2_u 原本使用 mixu_plab_sub 作为目标，因此输出级 DBM loss 也保留
                # 双头各自的伪标签，避免强行把 head2 拉向 head1 的标签。
                mixu_plab_sub_filtered = mixu_plab_sub.clone()
                mixu_plab_sub_filtered[conf_mixu < args.cpam_consistency_threshold] = num_classes

            if iter_num % 200 == 0:
                filt_ratio = (mixu_plab_filtered == num_classes).float().mean().item()
                writer.add_scalar('train/cpam_ignore_ratio', filt_ratio, iter_num)

            # ── CPAM ─────────────────────────────────────────────────────
            loss_prototype = loss_compact = loss_diversity = torch.tensor(0.0).cuda()

            if args.use_prototype and iter_num > args.proto_warmup:
                pf1_l = proj_head1(feat1_l); pf2_l = proj_head2(feat2_l)
                pf1_u = proj_head1(feat1_u); pf2_u = proj_head2(feat2_u)
                prob1_all = torch.cat([F.softmax(out1_l,dim=1), F.softmax(out1_u,dim=1)])
                prob2_all = torch.cat([F.softmax(out2_l,dim=1), F.softmax(out2_u,dim=1)])
                pf1_all   = torch.cat([pf1_l, pf1_u])
                pf2_all   = torch.cat([pf2_l, pf2_u])

                with torch.no_grad():
                    proto1, proto2, shared = proto_memory.update_with_consistency(
                        prob1_all, prob2_all, pf1_all, pf2_all,
                        consistency_threshold=args.cpam_consistency_threshold,
                        conf_threshold=args.conf_thresh)

                if proto_memory.is_ready():
                    c1l, _ = F.softmax(out1_l,dim=1).max(1)
                    c2l, _ = F.softmax(out2_l,dim=1).max(1)
                    c1u, _ = F.softmax(out1_u,dim=1).max(1)
                    c2u, _ = F.softmax(out2_u,dim=1).max(1)

                    loss_prototype = (
                        0.7 * loss_proto_contrast(
                            pf1_l, pf2_l, proto1, proto2, shared,
                            mixl_lab, c1l, c2l) +
                        0.3 * loss_proto_contrast(
                            pf1_u, pf2_u, proto1, proto2, shared,
                            mixu_plab_filtered,          # ← Bug2修复：使用过滤后标签
                            c1u, c2u))

                    loss_compact, compact_info = loss_discrepancy_compact(
                        pf1_all, pf2_all, proto1, proto2, shared,
                        torch.cat([mixl_lab, mixu_plab_filtered]),  # ← Bug2修复
                        iter_num, args.train_iterations)

                    loss_diversity = prototype_diversity_regularization(proto1, proto2, margin=0.3)

                    if iter_num % 50 == 0:
                        writer.add_scalar('CPAM/contrast',  loss_prototype.item(), iter_num)
                        writer.add_scalar('CPAM/compact',   loss_compact.item(),   iter_num)
                        writer.add_scalar('CPAM/diversity', loss_diversity.item(), iter_num)

            proto_w   = get_warmup_weight(iter_num, args.proto_warmup, args.proto_weight)   if args.use_prototype else 0.0
            compact_w = get_warmup_weight(iter_num, args.proto_warmup, args.proto_compact_weight) if args.use_prototype else 0.0
            div_w     = get_warmup_weight(iter_num, args.proto_warmup, args.proto_div_weight)     if args.use_prototype else 0.0

            # ── DecoderBoundaryModule（BADFC + BTNA + output-level CE/Dice）────
            loss_db_badfc_l = loss_db_btna_l = torch.tensor(0.0).cuda()
            loss_db_badfc_u = loss_db_btna_u = torch.tensor(0.0).cuda()
            loss_db_ce_l    = loss_db_dice_l = torch.tensor(0.0).cuda()
            loss_db_ce_u    = loss_db_dice_u = torch.tensor(0.0).cuda()

            if args.use_decoder_boundary and db_module and soft_l is not None:
                # 有标签路径：原 DBM feature-level loss
                loss_db_badfc_l, loss_db_btna_l, info_dbl = db_module.forward_labeled(
                    df1_l, df2_l, mixl_lab, soft_l, iter_num)

                # v8.4：新增 output-level 边界 CE/Dice，直接监督 out1/out2。
                # v8.5：改用 class-adaptive alpha（Myo 加权更强）。
                if args.db_use_output_loss:
                    # ── 有标签路径：class-adaptive alpha ─────────────────
                    if _class_alphas is not None:
                        # 优化 1：逐像素 alpha_map，Myo 边界获得更强监督
                        alpha_map_l = build_alpha_map(
                            mixl_lab, _class_alphas, num_classes).detach()
                        bnd_weight_l = build_boundary_spatial_weight(
                            soft_l, mixl_lab, _class_alphas, num_classes)
                        loss_db_ce_l = 0.5 * (
                            boundary_weighted_ce_loss(
                                out1_l, mixl_lab, soft_l,
                                reliability=None, ignore_index=None,
                                alpha_map=alpha_map_l)
                            + boundary_weighted_ce_loss(
                                out2_l, mixl_lab, soft_l,
                                reliability=None, ignore_index=None,
                                alpha_map=alpha_map_l)
                        )
                    else:
                        # 退回全局标量 alpha（向后兼容）
                        bnd_weight_l = (1.0 + args.db_boundary_alpha * soft_l).detach()
                        loss_db_ce_l = 0.5 * (
                            boundary_weighted_ce_loss(
                                out1_l, mixl_lab, soft_l,
                                reliability=None, ignore_index=None,
                                alpha=args.db_boundary_alpha)
                            + boundary_weighted_ce_loss(
                                out2_l, mixl_lab, soft_l,
                                reliability=None, ignore_index=None,
                                alpha=args.db_boundary_alpha)
                        )
                    loss_db_dice_l = 0.5 * (
                        masked_soft_dice_loss(
                            out1_l, mixl_lab, bnd_weight_l,
                            num_classes=num_classes, reliability=None,
                            ignore_index=None, include_background=False)
                        + masked_soft_dice_loss(
                            out2_l, mixl_lab, bnd_weight_l,
                            num_classes=num_classes, reliability=None,
                            ignore_index=None, include_background=False)
                    )

                # 无标签路径：feature-level DBM 仍使用过滤后的 pseudo label，
                # 但 db_soft_u 来自 raw pseudo 的 SDF，避免 ignore 洞制造假边界。
                if db_soft_u is not None:
                    loss_db_badfc_u, loss_db_btna_u, info_dbu = db_module.forward_unlabeled(
                        df1_u, df2_u, mixu_plab_for_sdf, db_soft_u, db_rel_u,
                        F.softmax(out1_u, dim=1), F.softmax(out2_u, dim=1), iter_num)

                    if args.db_use_output_loss:
                        # ── 优化 2：无标签路径使用独立低阈值（v8.5）────────
                        # adap_w 用于 CE（conf_thresh=0.72），对噪声敏感；
                        # DBM-Dice 有 soft_boundary 加持，鲁棒性更高，
                        # 用 db_dice_conf_thresh=0.45 让更多像素贡献梯度。
                        adap_w_for_db = adaptive_confidence_weighting(
                            conf_mixu,
                            threshold=args.db_dice_conf_thresh,
                            temperature=args.temperature,
                            use_soft=bool(args.use_soft_threshold),
                        )
                        db_u_weight = (db_rel_u * adap_w_for_db).clamp(0.0, 1.0)

                        # ── 优化 1：无标签路径 class-adaptive alpha ──────
                        if _class_alphas is not None:
                            alpha_map_u = build_alpha_map(
                                mixu_plab_filtered, _class_alphas,
                                num_classes, ignore_index=num_classes).detach()
                            alpha_map_u_sub = build_alpha_map(
                                mixu_plab_sub_filtered, _class_alphas,
                                num_classes, ignore_index=num_classes).detach()
                            bnd_weight_u = build_boundary_spatial_weight(
                                db_soft_u, mixu_plab_filtered,
                                _class_alphas, num_classes,
                                ignore_index=num_classes)
                            bnd_weight_u_sub = build_boundary_spatial_weight(
                                db_soft_u, mixu_plab_sub_filtered,
                                _class_alphas, num_classes,
                                ignore_index=num_classes)
                            loss_db_ce_u = 0.5 * (
                                boundary_weighted_ce_loss(
                                    out1_u, mixu_plab_filtered, db_soft_u,
                                    reliability=db_u_weight,
                                    ignore_index=num_classes,
                                    alpha_map=alpha_map_u)
                                + boundary_weighted_ce_loss(
                                    out2_u, mixu_plab_sub_filtered, db_soft_u,
                                    reliability=db_u_weight,
                                    ignore_index=num_classes,
                                    alpha_map=alpha_map_u_sub)
                            )
                            loss_db_dice_u = 0.5 * (
                                masked_soft_dice_loss(
                                    out1_u, mixu_plab_filtered, bnd_weight_u,
                                    num_classes=num_classes,
                                    reliability=db_u_weight,
                                    ignore_index=num_classes,
                                    include_background=False)
                                + masked_soft_dice_loss(
                                    out2_u, mixu_plab_sub_filtered, bnd_weight_u_sub,
                                    num_classes=num_classes,
                                    reliability=db_u_weight,
                                    ignore_index=num_classes,
                                    include_background=False)
                            )
                        else:
                            # 退回全局标量 alpha（向后兼容）
                            bnd_weight_u = (1.0 + args.db_boundary_alpha * db_soft_u).detach()
                            loss_db_ce_u = 0.5 * (
                                boundary_weighted_ce_loss(
                                    out1_u, mixu_plab_filtered, db_soft_u,
                                    reliability=db_u_weight,
                                    ignore_index=num_classes,
                                    alpha=args.db_boundary_alpha)
                                + boundary_weighted_ce_loss(
                                    out2_u, mixu_plab_sub_filtered, db_soft_u,
                                    reliability=db_u_weight,
                                    ignore_index=num_classes,
                                    alpha=args.db_boundary_alpha)
                            )
                            loss_db_dice_u = 0.5 * (
                                masked_soft_dice_loss(
                                    out1_u, mixu_plab_filtered, bnd_weight_u,
                                    num_classes=num_classes,
                                    reliability=db_u_weight,
                                    ignore_index=num_classes,
                                    include_background=False)
                                + masked_soft_dice_loss(
                                    out2_u, mixu_plab_sub_filtered, bnd_weight_u,
                                    num_classes=num_classes,
                                    reliability=db_u_weight,
                                    ignore_index=num_classes,
                                    include_background=False)
                            )

                if iter_num % 50 == 0:
                    writer.add_scalar('DB/badfc_l',   loss_db_badfc_l.item(), iter_num)
                    writer.add_scalar('DB/btna_l',    loss_db_btna_l.item(),  iter_num)
                    writer.add_scalar('DB/badfc_u',   loss_db_badfc_u.item(), iter_num)
                    writer.add_scalar('DB/btna_u',    loss_db_btna_u.item(),  iter_num)
                    writer.add_scalar('DB/output_ce_l',   loss_db_ce_l.item(),   iter_num)
                    writer.add_scalar('DB/output_dice_l', loss_db_dice_l.item(), iter_num)
                    writer.add_scalar('DB/output_ce_u',   loss_db_ce_u.item(),   iter_num)
                    writer.add_scalar('DB/output_dice_u', loss_db_dice_u.item(), iter_num)
                    if loss_db_badfc_l.item() > 0:
                        writer.add_scalar('DB/bnd_ratio', info_dbl.get('badfc_boundary_ratio', 0), iter_num)
                        writer.add_scalar('DB/aniso_l',   info_dbl.get('btna_aniso', 0), iter_num)
                    if loss_db_btna_u.item() > 0:
                        writer.add_scalar('DB/cd_entropy', info_dbu.get('btna_cd_entropy', 0), iter_num)
                    # v8.5：监控无标签 DBM-Dice 的有效像素比例（低阈值后应显著提升）
                    if db_soft_u is not None and args.db_use_output_loss:
                        dice_u_active = (db_u_weight > 0.1).float().mean().item()
                        writer.add_scalar('DB/dice_u_active_ratio', dice_u_active, iter_num)

            # ── 总损失 ────────────────────────────────────────────────────
            # C3: DBM 渐入权重，避免随机投影头 + 大初始 InfoNCE 冲击训练
            db_ramp = get_db_ramp(iter_num, args.db_start_iter, args.db_warmup_iters)
            if iter_num % 200 == 0 and args.use_decoder_boundary:
                writer.add_scalar('DB/ramp', db_ramp, iter_num)

            loss = (loss_l + loss_u + loss_dis
                    + proto_w   * loss_prototype
                    + compact_w * loss_compact
                    + div_w     * loss_diversity
                    + db_ramp * args.db_badfc_labeled_weight   * loss_db_badfc_l
                    + db_ramp * args.db_badfc_unlabeled_weight * loss_db_badfc_u
                    + db_ramp * args.db_btna_labeled_weight    * loss_db_btna_l
                    + db_ramp * args.db_btna_unlabeled_weight  * loss_db_btna_u
                    # v8.4：DBM output-level loss，直接作用于 logits，目标是提升 Dice
                    + db_ramp * args.db_boundary_ce_l_weight   * loss_db_ce_l
                    + db_ramp * args.db_boundary_dice_l_weight * loss_db_dice_l
                    + db_ramp * args.db_boundary_ce_u_weight   * loss_db_ce_u
                    + db_ramp * args.db_boundary_dice_u_weight * loss_db_dice_u)

            optimizer.zero_grad()
            loss.backward()

            all_modules = [model]
            if args.use_prototype:                              all_modules += [proj_head1, proj_head2]
            if args.use_decoder_boundary and db_module:        all_modules.append(db_module)
            for m in all_modules:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)

            optimizer.step()

            lr_ = polynomial_lr_decay(args.base_lr, iter_num, args.train_iterations, args.lr_power)
            for pg in optimizer.param_groups: pg['lr'] = lr_

            # ── 日志 ──────────────────────────────────────────────────────
            writer.add_scalar('train/loss',   loss.item(),  iter_num)
            writer.add_scalar('train/loss_l', loss_l.item(), iter_num)
            writer.add_scalar('train/loss_u', loss_u.item(), iter_num)
            writer.add_scalar('train/lr',     lr_,          iter_num)
            writer.add_scalar('train/conf',   avg_conf,     iter_num)

            if iter_num % 50 == 0:
                logging.info(
                    f'[{iter_num}/{args.train_iterations}] '
                    f'loss={loss.item():.4f} '
                    f'l={loss_l.item():.3f} u={loss_u.item():.3f} '
                    f'badfc_l={loss_db_badfc_l.item():.3f} '
                    f'btna_l={loss_db_btna_l.item():.3f} '
                    f'db_dice_l={loss_db_dice_l.item():.3f} '
                    f'db_dice_u={loss_db_dice_u.item():.3f} '
                    f'db_ramp={db_ramp:.2f} '
                    f'conf={avg_conf:.3f} lr={lr_:.5f}')

            if iter_num % 200 == 1:
                writer.add_image('train/img_l', mixl_img[0, 0:1], iter_num)
                writer.add_image('train/img_u', mixu_img[0, 0:1], iter_num)
                pred_l = ((out1_l+out2_l)/2).argmax(1, keepdim=True)
                pred_u = ((out1_u+out2_u)/2).argmax(1, keepdim=True)
                writer.add_image('train/pred_l', pred_l[0]*50, iter_num)
                writer.add_image('train/pred_u', pred_u[0]*50, iter_num)
                writer.add_image('train/gt_l',   mixl_lab[0].unsqueeze(0)*50, iter_num)
                if soft_l is not None:
                    writer.add_image('train/soft_boundary_l', soft_l[0].unsqueeze(0), iter_num)

            # ── 验证 ──────────────────────────────────────────────────────
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                if args.use_prototype: proj_head1.eval(); proj_head2.eval()
                if args.use_decoder_boundary and db_module: db_module.eval()

                ml = 0.0
                with torch.no_grad():
                    for vb in valloader:
                        ml += np.array(val_2d.test_single_volume(
                            vb["image"], vb["label"], model, classes=num_classes))
                ml /= len(db_val)

                for ci in range(num_classes - 1):
                    writer.add_scalar(f'val/class{ci+1}_dice', ml[ci, 0], iter_num)
                    writer.add_scalar(f'val/class{ci+1}_hd95', ml[ci, 1], iter_num)

                perf = np.mean(ml, axis=0)[0]
                writer.add_scalar('val/mean_dice', perf, iter_num)

                if perf > best_performance:
                    best_performance = perf
                    for fp in [
                        os.path.join(snapshot_path, f'{args.model}_best_model.pth'),
                        os.path.join(snapshot_path, f'iter_{iter_num}_dice_{perf:.4f}.pth'),
                    ]:
                        save_checkpoint(model, optimizer, fp,
                                        proj_head1, proj_head2, db_module)

                logging.info(f'[Val] {iter_num}  dice={perf:.4f}  best={best_performance:.4f}')

                model.train()
                if args.use_prototype: proj_head1.train(); proj_head2.train()
                if args.use_decoder_boundary and db_module: db_module.train()

            if iter_num >= args.train_iterations: break
        if iter_num >= args.train_iterations: break

    writer.close()
    logging.info(f"Training done. Best dice={best_performance:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = get_parser().parse_args()

    if args.deterministic:
        cudnn.benchmark = False; cudnn.deterministic = True
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)

    pre_path   = f"./model/CML/ACDC_{args.exp}_{args.labelnum}_labeled/pre_train"
    train_path = f"./model/CML/ACDC_{args.exp}_{args.labelnum}_labeled/train"
    for p_ in [pre_path, train_path]: os.makedirs(p_, exist_ok=True)

    logging.basicConfig(filename=train_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    print("\n" + "=" * 65)
    print("CML v8.3  Dice-bug-fixed: 置信度乘法权重 + CPAM 过滤伪标签")
    print("=" * 65)

    pre_train(args, pre_path)
    train(args, pre_path, train_path)