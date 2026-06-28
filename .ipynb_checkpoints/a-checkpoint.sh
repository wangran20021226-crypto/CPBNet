#!/bin/bash
# ════════════════════════════════════════════════════════════════════════
# Run 4: CPAM 权重重平衡(主对比加强 + 温度上调)
# 假设: CPAM 主对比项有更大潜力可挖;diversity 项在 4 类下贡献小
# 改动:
#   proto_weight:          0.60 → 0.80(主对比加强)
#   proto_compact_weight:  0.25 → 0.30(紧凑略加)
#   proto_div_weight:      0.08 → 0.05(diversity 在 4 类下贡献小,降权)
#   proto_temp:            0.07 → 0.10(温度略升,梯度更平滑)
# 预期: dice +0.10 ~ +0.20
# 风险: temp 升高可能让对比信号变弱,需要看 CPAM/contrast 损失是否还在下降
# ════════════════════════════════════════════════════════════════════════

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
  --conf_thresh 0.72 \
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
  --cpam_consistency_threshold 0.75 \
  \
  --use_decoder_boundary 1 \
  --db_start_iter 5000 \
  --db_warmup_iters 2000 \
  --db_proj_dim 64 \
  --db_temp_boundary 0.1 \
  --db_temp_inner 0.2 \
  --db_aniso_weight 1.0 \
  --db_cd_aniso_weight 0.5 \
  --db_cd_entropy_weight 0.3 \
  --db_badfc_labeled_weight 0.3 \
  --db_badfc_unlabeled_weight 0.15 \
  --db_btna_labeled_weight 0.15 \
  --db_btna_unlabeled_weight 0.08 \
  --db_consistency_threshold 0.6 \
  \
  --exp Run4_cpam_rebalance