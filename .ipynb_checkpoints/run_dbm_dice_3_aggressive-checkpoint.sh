#!/bin/bash
# ============================================================
# DBM-Dice 参数组 3：增强版（v8.5）
# 目标：更强地推动 Dice 提升。
# v8.5 新增：
#   - class-adaptive alpha（Myo 边界加权 4.0，RV 2.0，LV 1.5，BG 0.5）
#   - 无标签 DBM-Dice 独立低阈值（db_dice_conf_thresh=0.45）
# ============================================================

python CML_ACDC_train.py \
  --labelnum 7 \
  --batch_size 24 \
  --labeled_bs 12 \
  --base_lr 0.01 \
  --pre_iterations 10000 \
  --train_iterations 30000 \
  --l_weight 1.0 \
  --u_weight 0.6 \
  --dis_weight 0.1 \
  --mask_ratio 0.4 \
  --temperature 0.5 \
  --conf_thresh 0.70 \
  --conf_weight 1.0 \
  --use_soft_threshold 1 \
  --lr_power 0.9 \
  --rampup_length 0.3 \
  --use_prototype 1 \
  --proto_dim 128 \
  --proto_weight 0.80 \
  --proto_compact_weight 0.30 \
  --proto_div_weight 0.05 \
  --proto_momentum 0.99 \
  --proto_temp 0.10 \
  --proto_warmup 2000 \
  --use_noise_contrast 0 \
  --cpam_consistency_threshold 0.72 \
  \
  --use_decoder_boundary 1 \
  --db_start_iter 2000 \
  --db_warmup_iters 2500 \
  --db_proj_dim 64 \
  --db_temp_boundary 0.10 \
  --db_temp_inner 0.20 \
  --db_aniso_weight 1.0 \
  --db_cd_aniso_weight 0.5 \
  --db_cd_entropy_weight 0.3 \
  --db_badfc_labeled_weight 0.25 \
  --db_badfc_unlabeled_weight 0.10 \
  --db_btna_labeled_weight 0.10 \
  --db_btna_unlabeled_weight 0.04 \
  --db_consistency_threshold 0.65 \
  \
  --db_use_output_loss 1 \
  --db_boundary_alpha 2.5 \
  --db_boundary_ce_l_weight 0.15 \
  --db_boundary_dice_l_weight 0.30 \
  --db_boundary_ce_u_weight 0.06 \
  --db_boundary_dice_u_weight 0.12 \
  \
  --db_use_class_alpha 1 \
  --db_alpha_bg   0.5 \
  --db_alpha_rv   2.0 \
  --db_alpha_myo  4.0 \
  --db_alpha_lv   1.5 \
  --db_dice_conf_thresh 0.45 \
  \
  --exp Run8E_DBM_ClassAlpha_LowThresh