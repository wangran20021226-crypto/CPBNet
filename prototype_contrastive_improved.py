

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




def get_scheduled_temperature(iter_num: int, 
                              max_iter: int,
                              init_temp: float = 0.2, 
                              final_temp: float = 0.07) -> float:
    """温度退火调度"""
    if max_iter <= 0:
        return float(init_temp)
    progress = min(float(iter_num) / float(max_iter), 1.0)
    return float(init_temp * (1.0 - progress) + final_temp * progress)
