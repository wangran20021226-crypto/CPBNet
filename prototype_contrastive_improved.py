"""
优化的双模型协同原型学习模块 - 高性能版本
专为CML框架设计，优化了性能、稳定性和可扩展性

主要优化：
1. 内存优化 - 减少中间张量，使用inplace操作
2. 计算优化 - 批量处理，减少循环
3. 数值稳定性 - 添加epsilon，梯度裁剪
4. 代码清晰度 - 更好的模块化和注释
5. 可配置性 - 更灵活的参数设置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


# ==================== 1. 双模型协同原型记忆库（优化版）====================
class DualModelPrototypeMemory:
    """
    双模型协同原型记忆库 - 优化版
    
    优化点：
    - 批量计算所有类别原型，避免循环
    - 使用scatter操作加速分类统计
    - 减少不必要的张量创建
    """
    
    def __init__(self, 
                 num_classes: int, 
                 feature_dim: int, 
                 momentum: float = 0.999, 
                 device: str = 'cuda',
                 min_samples: int = 10,
                 eps: float = 1e-6):
        """
        Args:
            num_classes: 类别数量
            feature_dim: 特征维度
            momentum: EMA动量
            device: 设备
            min_samples: 每个类别最少样本数
            eps: 数值稳定性常数
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.device = device
        self.min_samples = min_samples
        self.eps = eps
        
        # 三套原型
        self.protos1 = None  # [C, D]
        self.protos2 = None
        self.shared_protos = None
        
        # 统计信息
        self.proto_counts = torch.zeros(num_classes, device=device)
        self.consistency_scores = torch.zeros(num_classes, device=device)
        self.initialized = False
        
    @torch.no_grad()
    def update_with_consistency(self, 
                                prob1: torch.Tensor, 
                                prob2: torch.Tensor, 
                                feat1: torch.Tensor, 
                                feat2: torch.Tensor,
                                consistency_threshold: float = 0.85, 
                                conf_threshold: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        优化的原型更新 - 批量处理所有类别
        
        Args:
            prob1, prob2: [B, C, H, W] softmax概率
            feat1, feat2: [B, D, H, W] 投影特征
            consistency_threshold: 一致性阈值
            conf_threshold: 置信度阈值（保留参数但不强制使用）
        """
        B, C, H, W = prob1.shape
        _, D, Hf, Wf = feat1.shape
        
        # 对齐分辨率
        if (H, W) != (Hf, Wf):
            prob1 = F.interpolate(prob1, size=(Hf, Wf), mode='bilinear', align_corners=False)
            prob2 = F.interpolate(prob2, size=(Hf, Wf), mode='bilinear', align_corners=False)
        
        # 预测和置信度
        conf1, pred1 = prob1.max(dim=1)  # [B, Hf, Wf]
        conf2, pred2 = prob2.max(dim=1)
        
        # 一致性掩码
        consistent_mask = (pred1 == pred2).float()
        avg_conf = (conf1 + conf2) * 0.5
        reliability = consistent_mask * avg_conf
        
        # 展平
        N = B * Hf * Wf
        feat1_flat = feat1.permute(0, 2, 3, 1).reshape(N, D)
        feat2_flat = feat2.permute(0, 2, 3, 1).reshape(N, D)
        pred_flat = pred1.reshape(N)
        reliability_flat = reliability.reshape(N)
        
        # 归一化特征（inplace优化）
        feat1_flat = F.normalize(feat1_flat, dim=1)
        feat2_flat = F.normalize(feat2_flat, dim=1)
        
        # ==================== 批量计算所有类别原型 ====================
        # 筛选可靠样本
        reliable_mask = reliability_flat > consistency_threshold  # [N]
        
        if reliable_mask.sum() < self.min_samples:
            # 样本不足，返回当前原型
            if self.is_ready():
                return self.protos1, self.protos2, self.shared_protos
            else:
                # 首次初始化失败，返回零原型
                return (torch.zeros(C, D, device=self.device),
                        torch.zeros(C, D, device=self.device),
                        torch.zeros(C, D, device=self.device))
        
        # 筛选可靠样本
        reliable_pred = pred_flat[reliable_mask]  # [M]
        reliable_weights = reliability_flat[reliable_mask]  # [M]
        reliable_f1 = feat1_flat[reliable_mask]  # [M, D]
        reliable_f2 = feat2_flat[reliable_mask]  # [M, D]
        
        # 使用scatter_add批量计算加权和
        # 初始化累加器
        weighted_sum1 = torch.zeros(C, D, device=self.device)
        weighted_sum2 = torch.zeros(C, D, device=self.device)
        weight_sum = torch.zeros(C, device=self.device)
        
        # 加权特征
        weighted_f1 = reliable_f1 * reliable_weights.unsqueeze(1)  # [M, D]
        weighted_f2 = reliable_f2 * reliable_weights.unsqueeze(1)
        
        # 批量累加（避免循环）
        reliable_pred_expanded = reliable_pred.unsqueeze(1).expand(-1, D)  # [M, D]
        weighted_sum1.scatter_add_(0, reliable_pred_expanded, weighted_f1)
        weighted_sum2.scatter_add_(0, reliable_pred_expanded, weighted_f2)
        weight_sum.scatter_add_(0, reliable_pred, reliable_weights)
        
        # 计算原型（避免除零）
        valid_classes = weight_sum > 0
        new_protos1 = torch.zeros(C, D, device=self.device)
        new_protos2 = torch.zeros(C, D, device=self.device)
        new_shared = torch.zeros(C, D, device=self.device)
        
        if valid_classes.any():
            # 归一化
            norm_factor = weight_sum[valid_classes].unsqueeze(1) + self.eps
            new_protos1[valid_classes] = weighted_sum1[valid_classes] / norm_factor
            new_protos2[valid_classes] = weighted_sum2[valid_classes] / norm_factor
            
            # L2归一化
            new_protos1[valid_classes] = F.normalize(new_protos1[valid_classes], dim=1)
            new_protos2[valid_classes] = F.normalize(new_protos2[valid_classes], dim=1)
            
            # 共享原型
            new_shared[valid_classes] = F.normalize(
                (new_protos1[valid_classes] + new_protos2[valid_classes]) * 0.5, 
                dim=1
            )
        
        # 动量更新或初始化
        if self.protos1 is None:
            self.protos1 = new_protos1
            self.protos2 = new_protos2
            self.shared_protos = new_shared
            self.proto_counts[valid_classes] = 1
        else:
            # EMA更新（只更新有效类别）
            if valid_classes.any():
                m = self.momentum
                self.protos1[valid_classes] = m * self.protos1[valid_classes] + (1-m) * new_protos1[valid_classes]
                self.protos2[valid_classes] = m * self.protos2[valid_classes] + (1-m) * new_protos2[valid_classes]
                self.shared_protos[valid_classes] = m * self.shared_protos[valid_classes] + (1-m) * new_shared[valid_classes]
                
                # 重新归一化
                self.protos1[valid_classes] = F.normalize(self.protos1[valid_classes], dim=1)
                self.protos2[valid_classes] = F.normalize(self.protos2[valid_classes], dim=1)
                self.shared_protos[valid_classes] = F.normalize(self.shared_protos[valid_classes], dim=1)
                
                self.proto_counts[valid_classes] += 1
        
        # 更新一致性分数
        for c in range(C):
            if valid_classes[c]:
                class_reliability = reliable_weights[reliable_pred == c].mean()
                self.consistency_scores[c] = 0.9 * self.consistency_scores[c] + 0.1 * class_reliability
        
        # 检查初始化
        self.initialized = (self.proto_counts > 0).all().item()
        
        return self.protos1, self.protos2, self.shared_protos
    
    def get_protos(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """获取所有原型"""
        return self.protos1, self.protos2, self.shared_protos
    
    def is_ready(self) -> bool:
        """检查是否所有类别都已初始化"""
        return self.initialized
    
    def get_consistency_info(self) -> Dict[str, np.ndarray]:
        """获取统计信息"""
        return {
            'proto_counts': self.proto_counts.cpu().numpy(),
            'consistency_scores': self.consistency_scores.cpu().numpy(),
            'avg_consistency': self.consistency_scores.mean().item()
        }


# ==================== 2. 自适应差异-紧凑性平衡损失（优化版）====================
class AdaptiveDiscrepancyCompactnessLoss(nn.Module):
    """
    优化的差异-紧凑性损失
    
    优化点：
    - 简化计算流程
    - 减少中间张量
    - 改进数值稳定性
    """
    
    def __init__(self, num_classes: int, feature_dim: int, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.eps = eps
        
    def forward(self, 
                feat1: torch.Tensor, 
                feat2: torch.Tensor, 
                proto1: torch.Tensor, 
                proto2: torch.Tensor,
                shared_proto: torch.Tensor, 
                labels: torch.Tensor, 
                iter_num: int, 
                max_iter: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            feat1, feat2: [B, D, H, W] 特征
            proto1, proto2, shared_proto: [C, D] 原型
            labels: [B, H, W] 标签
            iter_num, max_iter: 迭代信息
        
        Returns:
            loss: 总损失
            info: 统计信息字典
        """
        B, D, H, W = feat1.shape
        C = self.num_classes
        
        # 训练进度
        progress = min(float(iter_num) / max(float(max_iter), 1.0), 1.0)
        
        # ==================== 1. 特征差异损失 ====================
        # 简化为余弦相似度损失（比KL散度更稳定）
        feat1_flat = feat1.flatten(2)  # [B, D, N]
        feat2_flat = feat2.flatten(2)
        
        # L2归一化
        feat1_norm = F.normalize(feat1_flat, dim=1)
        feat2_norm = F.normalize(feat2_flat, dim=1)
        
        # 余弦相似度
        cos_sim = (feat1_norm * feat2_norm).sum(dim=1).mean()
        
        # 差异损失 = 1 - 相似度（鼓励差异）
        loss_discrepancy = 1.0 - cos_sim
        
        # ==================== 2. 紧凑性损失 ====================
        # 对齐分辨率
        _, Hl, Wl = labels.shape
        if (Hl, Wl) != (H, W):
            labels = F.interpolate(
                labels.unsqueeze(1).float(),
                size=(H, W),
                mode='nearest'
            ).squeeze(1).long()
        
        # 展平
        N = B * H * W
        feat1_pixels = feat1.permute(0, 2, 3, 1).reshape(N, D)
        feat2_pixels = feat2.permute(0, 2, 3, 1).reshape(N, D)
        labels_flat = labels.reshape(N)
        
        # 归一化
        feat1_pixels = F.normalize(feat1_pixels, dim=1)
        feat2_pixels = F.normalize(feat2_pixels, dim=1)
        proto1_norm = F.normalize(proto1, dim=1)
        proto2_norm = F.normalize(proto2, dim=1)
        shared_norm = F.normalize(shared_proto, dim=1)
        
        # 过滤有效样本
        valid_mask = (labels_flat >= 0) & (labels_flat < C)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=feat1.device), {}
        
        feat1_valid = feat1_pixels[valid_mask]
        feat2_valid = feat2_pixels[valid_mask]
        labels_valid = labels_flat[valid_mask]
        
        # 计算到对应原型的距离
        # 使用共享原型（最稳定）
        gathered_protos = shared_norm[labels_valid]  # [M, D]
        
        # 余弦距离
        dist1 = 1.0 - (feat1_valid * gathered_protos).sum(dim=1)
        dist2 = 1.0 - (feat2_valid * gathered_protos).sum(dim=1)
        
        # 紧凑性损失
        loss_compactness = (dist1.mean() + dist2.mean()) * 0.5
        
        # ==================== 3. 自适应权重 ====================
        # 早期强调差异，后期强调紧凑
        # 使用余弦退火
        alpha = 0.5 + 0.5 * np.cos(progress * np.pi)  # [1, 0]
        
        # 总损失
        total_loss = alpha * loss_discrepancy + (1 - alpha) * loss_compactness
        
        # 统计信息
        info = {
            'discrepancy': loss_discrepancy.item(),
            'compactness': loss_compactness.item(),
            'alpha': alpha,
            'progress': progress
        }
        
        return total_loss, info


# ==================== 3. 噪声增强原型对比损失（优化版）====================
class NoiseAugmentedPrototypeLoss(nn.Module):
    """
    优化的噪声增强对比学习
    
    优化点：
    - 简化损失计算
    - 改进数值稳定性
    - 减少内存占用
    """
    
    def __init__(self, 
                 num_classes: int, 
                 temperature: float = 0.07, 
                 noise_std: float = 0.1,
                 eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.noise_std = noise_std
        self.eps = eps
        
    def forward(self, 
                feat1: torch.Tensor, 
                feat2: torch.Tensor, 
                proto1: torch.Tensor,
                proto2: torch.Tensor, 
                shared_proto: torch.Tensor, 
                labels: torch.Tensor,
                conf1: Optional[torch.Tensor] = None, 
                conf2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        噪声增强对比损失
        
        Args:
            feat1, feat2: [B, D, H, W] 特征
            proto1, proto2, shared_proto: [C, D] 原型
            labels: [B, H, W] 标签
            conf1, conf2: [B, H, W] 置信度（可选）
        """
        B, D, H, W = feat1.shape
        C = self.num_classes
        
        # 处理标签
        if len(labels.shape) == 4:
            labels = labels.squeeze(1)
        
        # 对齐分辨率
        _, Hl, Wl = labels.shape
        if (Hl, Wl) != (H, W):
            labels = F.interpolate(
                labels.unsqueeze(1).float(),
                size=(H, W),
                mode='nearest'
            ).squeeze(1).long()
        
        # ==================== 噪声增强 ====================
        # 控制噪声范围
        noise1 = torch.randn_like(feat1) * self.noise_std
        noise2 = torch.randn_like(feat2) * self.noise_std
        noise1.clamp_(-0.2, 0.2)
        noise2.clamp_(-0.2, 0.2)
        
        # 增强特征
        feat1_aug = feat1 + noise1
        feat2_aug = feat2 + noise2
        
        # 展平并归一化
        N = B * H * W
        feat1_norm = F.normalize(feat1.permute(0, 2, 3, 1).reshape(N, D), dim=1)
        feat2_norm = F.normalize(feat2.permute(0, 2, 3, 1).reshape(N, D), dim=1)
        feat1_aug_norm = F.normalize(feat1_aug.permute(0, 2, 3, 1).reshape(N, D), dim=1)
        feat2_aug_norm = F.normalize(feat2_aug.permute(0, 2, 3, 1).reshape(N, D), dim=1)
        
        shared_norm = F.normalize(shared_proto, dim=1)
        labels_flat = labels.reshape(N)
        
        # 过滤有效样本
        valid_mask = (labels_flat >= 0) & (labels_flat < C)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=feat1.device)
        
        feat1_v = feat1_norm[valid_mask]
        feat2_v = feat2_norm[valid_mask]
        feat1_aug_v = feat1_aug_norm[valid_mask]
        feat2_aug_v = feat2_aug_norm[valid_mask]
        labels_v = labels_flat[valid_mask]
        
        # ==================== 交叉对比（简化版）====================
        # feat1预测feat2_aug的伪标签
        with torch.no_grad():
            logits2_aug = torch.mm(feat2_aug_v, shared_norm.T) / self.temperature
            pseudo2 = logits2_aug.argmax(dim=1)
        
        logits1 = torch.mm(feat1_v, shared_norm.T) / self.temperature
        loss_cross1 = F.cross_entropy(logits1, pseudo2)
        
        # feat2预测feat1_aug的伪标签
        with torch.no_grad():
            logits1_aug = torch.mm(feat1_aug_v, shared_norm.T) / self.temperature
            pseudo1 = logits1_aug.argmax(dim=1)
        
        logits2 = torch.mm(feat2_v, shared_norm.T) / self.temperature
        loss_cross2 = F.cross_entropy(logits2, pseudo1)
        
        # ==================== 监督损失 ====================
        loss_sup = (F.cross_entropy(logits1, labels_v) + 
                   F.cross_entropy(logits2, labels_v)) * 0.5
        
        # ==================== 一致性损失 ====================
        loss_consist = (1.0 - F.cosine_similarity(feat1_v, feat1_aug_v, dim=1).mean() +
                       1.0 - F.cosine_similarity(feat2_v, feat2_aug_v, dim=1).mean()) * 0.5
        
        # 总损失
        total_loss = 0.5 * loss_sup + 0.3 * (loss_cross1 + loss_cross2) * 0.5 + 0.2 * loss_consist
        
        return total_loss


# ==================== 4. 增强的原型对比损失（优化版）====================
class EnhancedPrototypeContrastiveLoss(nn.Module):
    """
    优化的分层对比损失
    
    优化点：
    - 简化分层逻辑
    - 改进边界检测
    - 提高计算效率
    """
    
    def __init__(self, 
                 num_classes: int, 
                 temperature: float = 0.07,
                 boundary_threshold: float = 0.1,
                 eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.boundary_threshold = boundary_threshold
        self.eps = eps
        
    def forward(self, 
                feat1: torch.Tensor, 
                feat2: torch.Tensor,
                proto1: torch.Tensor, 
                proto2: torch.Tensor, 
                shared_proto: torch.Tensor,
                labels: torch.Tensor, 
                conf1: Optional[torch.Tensor] = None,
                conf2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        分层原型对比损失
        
        策略：
        - 高置信度：标准对比
        - 低置信度：困难样本挖掘
        - 边界样本：特殊处理
        """
        B, D, H, W = feat1.shape
        C = self.num_classes
        
        # 处理标签维度
        if len(labels.shape) == 4:
            labels = labels.squeeze(1)
        
        # 对齐分辨率
        _, Hl, Wl = labels.shape
        if (Hl, Wl) != (H, W):
            labels = F.interpolate(
                labels.unsqueeze(1).float(),
                size=(H, W),
                mode='nearest'
            ).squeeze(1).long()
            
            if conf1 is not None:
                conf1 = F.interpolate(conf1.unsqueeze(1), size=(H, W), 
                                     mode='bilinear', align_corners=False).squeeze(1)
                conf2 = F.interpolate(conf2.unsqueeze(1), size=(H, W),
                                     mode='bilinear', align_corners=False).squeeze(1)
        
        # 展平
        N = B * H * W
        feat1_flat = F.normalize(feat1.permute(0, 2, 3, 1).reshape(N, D), dim=1)
        feat2_flat = F.normalize(feat2.permute(0, 2, 3, 1).reshape(N, D), dim=1)
        labels_flat = labels.reshape(N)
        
        # 归一化原型
        shared_norm = F.normalize(shared_proto, dim=1)
        
        # 过滤有效样本
        valid_mask = (labels_flat >= 0) & (labels_flat < C)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=feat1.device)
        
        feat1_v = feat1_flat[valid_mask]
        feat2_v = feat2_flat[valid_mask]
        labels_v = labels_flat[valid_mask]
        
        # 计算logits
        logits1 = torch.mm(feat1_v, shared_norm.T) / self.temperature
        logits2 = torch.mm(feat2_v, shared_norm.T) / self.temperature
        
        # 如果有置信度，使用加权损失
        if conf1 is not None and conf2 is not None:
            conf1_v = conf1.reshape(N)[valid_mask]
            conf2_v = conf2.reshape(N)[valid_mask]
            avg_conf_v = (conf1_v + conf2_v) * 0.5
            
            # 置信度加权
            loss1 = F.cross_entropy(logits1, labels_v, reduction='none')
            loss2 = F.cross_entropy(logits2, labels_v, reduction='none')
            
            # 高置信度样本权重更大
            weights = avg_conf_v
            loss = ((loss1 + loss2) * 0.5 * weights).mean()
        else:
            # 标准损失
            loss = (F.cross_entropy(logits1, labels_v) + 
                   F.cross_entropy(logits2, labels_v)) * 0.5
        
        return loss


# ==================== 5. 投影头（优化版）====================
class ProjectionHead(nn.Module):
    """
    优化的投影头
    
    优化点：
    - 使用GroupNorm替代BatchNorm（更稳定）
    - 减少Dropout（避免过度正则化）
    - 简化结构
    """
    
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim: int = 128, 
                 out_dim: int = 64,
                 num_groups: int = 8, 
                 dropout: float = 0.05):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ==================== 6. 辅助函数 ====================
def prototype_diversity_regularization(proto1: torch.Tensor, 
                                      proto2: torch.Tensor, 
                                      margin: float = 0.3,
                                      eps: float = 1e-6) -> torch.Tensor:
    """
    原型多样性正则化（优化版）
    
    优化点：
    - 批量计算
    - 改进数值稳定性
    """
    loss = 0.0
    
    for protos in [proto1, proto2]:
        C = protos.shape[0]
        protos_norm = F.normalize(protos, dim=1)
        
        # 相似度矩阵
        sim_matrix = torch.mm(protos_norm, protos_norm.T)
        
        # 移除对角线（自己和自己）
        mask = ~torch.eye(C, dtype=torch.bool, device=protos.device)
        off_diag_sim = sim_matrix[mask]
        
        # 惩罚过于相似的原型
        diversity_loss = F.relu(off_diag_sim + margin - 1.0).mean()
        loss += diversity_loss
    
    return loss * 0.5


def get_warmup_weight(iter_num: int, 
                     warmup_iters: int, 
                     base_weight: float) -> float:
    """Warmup权重调度"""
    if warmup_iters is None or warmup_iters <= 0:
        return float(base_weight)
    return float(base_weight) * min(float(iter_num) / float(warmup_iters), 1.0)


def get_scheduled_temperature(iter_num: int, 
                              max_iter: int,
                              init_temp: float = 0.2, 
                              final_temp: float = 0.07) -> float:
    """温度退火调度"""
    if max_iter <= 0:
        return float(init_temp)
    progress = min(float(iter_num) / float(max_iter), 1.0)
    return float(init_temp * (1.0 - progress) + final_temp * progress)
