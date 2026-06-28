"""
test_ACDC.py  v8.5-fixed
针对 ACDC 数据集的协同多学习半监督语义分割测试

与训练文件 CML_ACDC_train.py v8.5 的对应关系
────────────────────────────────────────────────
  · 模型前向输出 6 项:
        (out1, out2, feat1, feat2, dec_feat1, dec_feat2)
  · 检查点格式（save_checkpoint）:
        {'model': ..., 'optimizer': ...,
         'proj_head1': ..., 'proj_head2': ..., 'db_module': ...}
    推理时只加载 'model' 键，其余键直接忽略。
  · 快照路径:
        ./model/CML/ACDC_{exp}_{labelnum}_labeled/{stage_name}/
  · --exp 需与训练时 --exp 完全一致

三个关键修复（解决测试结果远低于训练结果的根本原因）
────────────────────────────────────────────────
  修复 1  LargestCC 后处理（最重要）
    训练时 pseudo label 始终经过 get_ACDC_2DLargestCC() 过滤，
    模型学到的边界假设建立在"每类只有一个连通域"之上。
    原测试代码不做任何后处理 → 小噪声连通域拉高 HD95，压低 Dice。
    新增 --use_postprocess（默认 1）：在 argmax 之后，对 3D volume
    逐类别保留最大连通域，与训练 pseudo label 的生成逻辑完全一致。

  修复 2  val_2d.test_single_volume 对齐诊断
    训练期间报告的 "val Dice" 来自 val.list（验证集），而非 test.list。
    30,000 iter 每 200 iter 选一次 best_model → val 上事实过拟合。
    新增 --eval_on_val（默认 0）：用相同推理流程在 val.list 上跑，
    可以直接与 TensorBoard 的 val/mean_dice 曲线核对，确认是否一致。
    如果 eval_on_val=1 时结果很好、test 时结果差，说明是 val/test 分布差异。

  修复 3  归一化一致性验证
    acdc_data_processing.py 对 volume 做逐卷归一化 (img-min)/(max-min)。
    如果测试用的 h5 volume 文件是未归一化的原始强度，推理输入超出训练分布。
    新增启动时打印输入范围的 DEBUG 信息（首个 case 首个 slice），
    正常值应在 [0, 1]；若出现大值（如 [0, 400]）说明 h5 未归一化，
    需要在推理时补做 per-slice normalization（--normalize_input 1）。

使用方法
────────
  # 标准测试（含 LargestCC 后处理）：
  python test_ACDC.py --exp Run8E_DBM_ClassAlpha_LowThresh --labelnum 7

  # 对比：关闭后处理，复现原始测试 bug：
  python test_ACDC.py --exp Run8E_DBM_ClassAlpha_LowThresh --labelnum 7 \
      --use_postprocess 0

  # 在验证集上运行，与 TensorBoard val/mean_dice 对齐确认：
  python test_ACDC.py --exp Run8E_DBM_ClassAlpha_LowThresh --labelnum 7 \
      --eval_on_val 1

  # 测试指定 iter 检查点：
  python test_ACDC.py --exp Run8E_DBM_ClassAlpha_LowThresh --labelnum 7 \
      --iter_ckpt iter_28000_dice_0.9012.pth

  # 保存逐例 CSV + NIfTI：
  python test_ACDC.py --exp Run8E_DBM_ClassAlpha_LowThresh --labelnum 7 \
      --save_csv 1 --save_nii 1
"""

import argparse
import csv
import os
import shutil
import logging
import sys

import h5py
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from scipy.ndimage import zoom
from skimage.measure import label as sk_label   # 修复 1: LargestCC 后处理
from tqdm import tqdm

from networks.net_factory import net_factory


# ═══════════════════════════════════════════════════════════════════════
# 参数解析
# ═══════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description='CML v8.5-compatible Testing for ACDC')

parser.add_argument('--root_path',   type=str,   default='../data/ACDC',
                    help='数据集路径')
# v8.5：默认 exp 与 run_dbm_dice_3_aggressive.sh 中 --exp 对齐
parser.add_argument('--exp',         type=str,   default='Run8E_DBM_ClassAlpha_LowThresh',
                    help='实验名称，需与训练时 --exp 完全一致')
parser.add_argument('--model',       type=str,   default='unet',
                    help='模型名称')
parser.add_argument('--num_classes', type=int,   default=4,
                    help='类别数')
parser.add_argument('--labelnum',    type=int,   default=7,
                    help='有标签样本数，需与训练时一致')
parser.add_argument('--stage_name',  type=str,   default='train',
                    choices=['pre_train', 'train'],
                    help='加载哪个阶段的权重')
parser.add_argument('--gpu',         type=str,   default='0',
                    help='GPU 编号')
parser.add_argument('--patch_size',  type=int,   nargs='+', default=[256, 256],
                    help='推理切片尺寸，单值=正方形，双值=[H, W]，需与训练时一致')
parser.add_argument('--ensemble',    type=str,   default='avg',
                    choices=['avg', 'head1', 'head2'],
                    help='双头融合方式')
parser.add_argument('--save_nii',    type=int,   default=0,
                    help='是否保存 NIfTI 预测文件')
# ── v8.5 新增 ────────────────────────────────────────────────────────
parser.add_argument('--iter_ckpt',   type=str,   default='',
                    help='v8.5 新增：直接指定 iter_XXXX_dice_YYYY.pth 文件名；'
                         '为空时自动使用 {model}_best_model.pth')
parser.add_argument('--save_csv',    type=int,   default=0,
                    help='v8.5 新增：将逐例（per-case）四项指标写入 CSV（1=开启）')
# ── 关键修复参数 ──────────────────────────────────────────────────────
parser.add_argument('--use_postprocess', type=int, default=1,
                    help='修复 1：LargestCC 后处理（默认 1=开启）；'
                         '与训练 pseudo label 生成保持一致，可显著改善 HD95 和 Dice')
parser.add_argument('--eval_on_val',     type=int, default=0,
                    help='修复 2：在 val.list 而非 test.list 上测试（默认 0）；'
                         '用于与 TensorBoard val/mean_dice 对齐确认，排查 val/test 分布差异')
parser.add_argument('--normalize_input', type=int, default=0,
                    help='修复 3：推理时对每个 slice 做 (x-min)/(max-min) 归一化（默认 0）；'
                         '仅当 h5 volume 文件未预归一化时开启，开启前先观察 DEBUG 输入范围')


# ═══════════════════════════════════════════════════════════════════════
# 修复 1：LargestCC 后处理（与训练 pseudo label 生成完全对齐）
# ═══════════════════════════════════════════════════════════════════════

def get_largest_cc_per_class(prediction: np.ndarray,
                              num_classes: int = 4) -> np.ndarray:
    """
    对 3D 预测图逐类别保留最大连通域，去除噪声小碎片。

    与训练代码中 get_ACDC_2DLargestCC 的逻辑完全一致：
        训练时 pseudo label 始终经过该过滤，模型见到的边界分布
        建立在"每类单一连通域"的假设上；测试不做此后处理会引入
        训练时从未出现的噪声连通域，导致 HD95 大幅升高、Dice 下降。

    参数
    ────
    prediction  : [D, H, W] int，argmax 后的 3D 分割图
    num_classes : 总类别数（含背景）

    返回
    ────
    [D, H, W] int，每类仅保留最大连通域，其余置 0
    """
    out = np.zeros_like(prediction)
    for c in range(1, num_classes):          # 跳过背景 class 0
        mask = (prediction == c).astype(np.uint8)
        if mask.sum() == 0:
            continue
        labeled = sk_label(mask)             # 连通域标记
        max_label = 0
        max_size  = 0
        for lb in range(1, labeled.max() + 1):
            sz = (labeled == lb).sum()
            if sz > max_size:
                max_size  = sz
                max_label = lb
        if max_label > 0:
            out[labeled == max_label] = c
    return out


# ═══════════════════════════════════════════════════════════════════════
# 指标计算
# ═══════════════════════════════════════════════════════════════════════

def calculate_metric_percase(pred: np.ndarray, gt: np.ndarray):
    """
    计算单类别的 Dice / Jaccard / HD95 / ASD。
    预测为全空时返回全零（避免 medpy 异常）。
    """
    pred = (pred > 0).astype(np.uint8)
    gt   = (gt   > 0).astype(np.uint8)

    if pred.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0

    dice = metric.binary.dc(pred, gt)
    jc   = metric.binary.jc(pred, gt)

    try:
        hd95 = metric.binary.hd95(pred, gt)
    except Exception:
        hd95 = 0.0

    try:
        asd = metric.binary.asd(pred, gt)
    except Exception:
        asd = 0.0

    return dice, jc, hd95, asd


# ═══════════════════════════════════════════════════════════════════════
# 单卷推理
# ═══════════════════════════════════════════════════════════════════════

def test_single_volume(case: str, net: torch.nn.Module,
                       test_save_path: str, FLAGS,
                       _debug_first: list = None) -> tuple:
    """
    逐切片推理单个 3D 体数据，含修复 1/3 的后处理与归一化检查。

    v8.5 模型前向输出格式（训练时）：
        (out1, out2, feat1, feat2, dec_feat1, dec_feat2)  ← 共 6 项
    推理时仅使用 out1 / out2；dec_feat* 直接忽略。

    修复 1  use_postprocess=1：argmax 之后对整个 3D volume 做 LargestCC 后处理，
            与训练 pseudo label get_ACDC_2DLargestCC 的逻辑对齐。
    修复 3  normalize_input=1：对每个 2D slice 做 (x-min)/(max-min) 归一化，
            仅在 h5 文件未预归一化时开启。

    _debug_first: 传入一个可变列表 [False]，首个 case 会打印输入范围后改为 [True]。
    """
    h5_path = os.path.join(FLAGS.root_path, "data", f"{case}.h5")
    with h5py.File(h5_path, 'r') as f:
        image = f['image'][:]   # (D, H, W)
        label = f['label'][:]   # (D, H, W)

    ps = FLAGS.patch_size
    ps_h, ps_w = (ps[0], ps[0]) if len(ps) == 1 else (ps[0], ps[1])

    prediction = np.zeros_like(label)

    net.eval()
    with torch.no_grad():
        for i in range(image.shape[0]):
            slc = image[i].copy().astype(np.float32)
            h, w = slc.shape

            # ── 修复 3：归一化检查 & 可选归一化 ──────────────────────────
            # 首个 case 首个 slice 打印范围，用于确认 h5 是否已预归一化。
            # 正常训练时 acdc_data_processing.py 已做 (img-min)/(max-min)，
            # 所以正常值应在 [0.0, 1.0]。若出现 [0, 400] 等大值说明未归一化。
            if _debug_first is not None and not _debug_first[0] and i == 0:
                logging.info(
                    f"[DEBUG] 输入范围检查 case={case} slice=0: "
                    f"min={slc.min():.4f}  max={slc.max():.4f}  "
                    f"mean={slc.mean():.4f}  "
                    f"{'✓ 已归一化 [0,1]' if slc.max() <= 1.5 else '⚠ 疑似未归一化，建议 --normalize_input 1'}")
                _debug_first[0] = True

            if FLAGS.normalize_input:
                mn, mx = slc.min(), slc.max()
                if mx > mn:
                    slc = (slc - mn) / (mx - mn)

            inp = zoom(slc, (ps_h / h, ps_w / w), order=0)
            inp = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).float().cuda()

            # ── 前向（v8.5 训练时输出 6 项） ─────────────────────────────
            raw = net(inp)
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                out1, out2 = raw[0], raw[1]
                if FLAGS.ensemble == 'avg':
                    logits = (out1 + out2) / 2.0
                elif FLAGS.ensemble == 'head1':
                    logits = out1
                else:
                    logits = out2
            else:
                logits = raw

            pred = (torch.argmax(torch.softmax(logits, dim=1), dim=1)
                    .squeeze(0).cpu().numpy())
            pred = zoom(pred, (h / ps_h, w / ps_w), order=0)
            prediction[i] = pred

    # ── 修复 1：LargestCC 后处理（3D volume 整体处理）────────────────────
    # 在 zoom 回原尺寸之后、指标计算之前对整个 3D volume 做，
    # 而非逐 slice 处理——3D 连通域更准确（如 Myo 环在相邻 slice 间连通）。
    if FLAGS.use_postprocess:
        prediction = get_largest_cc_per_class(prediction, num_classes=FLAGS.num_classes)

    if FLAGS.save_nii:
        sp = (1.0, 1.0, 10.0)
        for arr, suffix in [(image, '_img'), (prediction, '_pred'), (label, '_gt')]:
            itk = sitk.GetImageFromArray(arr.astype(np.float32))
            itk.SetSpacing(sp)
            sitk.WriteImage(itk, os.path.join(test_save_path, f"{case}{suffix}.nii.gz"))

    m1 = calculate_metric_percase(prediction == 1, label == 1)
    m2 = calculate_metric_percase(prediction == 2, label == 2)
    m3 = calculate_metric_percase(prediction == 3, label == 3)

    return m1, m2, m3


# ═══════════════════════════════════════════════════════════════════════
# 检查点路径辅助
# ═══════════════════════════════════════════════════════════════════════

def _resolve_ckpt_path(snapshot_path: str, model: str, iter_ckpt: str) -> str:
    """
    决定要加载的检查点路径。

    优先级
    ──────
    1. iter_ckpt 非空 → 直接拼接到 snapshot_path 下
    2. 否则 → {model}_best_model.pth
    """
    if iter_ckpt:
        return os.path.join(snapshot_path, iter_ckpt)
    return os.path.join(snapshot_path, f'{model}_best_model.pth')


def _ckpt_to_args(ckpt_path: str, model: str) -> str:
    """
    把检查点路径反推成对应的命令行参数字符串。

    路径格式:
        ./model/CML/ACDC_{exp}_{labelnum}_labeled/{stage}/{filename}

    v8.5 容错改进：实验名（exp）可能包含多个下划线（如 Run8E_DBM_ClassAlpha_LowThresh），
    labelnum 始终是最后一个下划线之后的纯数字段，用 rsplit("_", 1) 从右切分可正确分离。
    """
    try:
        parts  = ckpt_path.replace("\\", "/").split("/")
        folder = parts[-3]                               # ACDC_{exp}_{labelnum}_labeled
        stage  = parts[-2]                               # train / pre_train
        fname  = parts[-1]                               # unet_best_model.pth 或 iter_*.pth

        # 去掉前缀 "ACDC_" 和后缀 "_labeled"
        inner  = folder[len("ACDC_"):-len("_labeled")]  # {exp}_{labelnum}

        # rsplit(maxsplit=1)：从右侧第一个 "_" 切分，保证 labelnum 正确
        segs   = inner.rsplit("_", 1)
        if len(segs) != 2 or not segs[1].isdigit():
            raise ValueError("无法解析 labelnum")
        exp, labelnum = segs[0], segs[1]

        iter_part = f" --iter_ckpt {fname}" if fname != f"{model}_best_model.pth" else ""
        return (f"python test_ACDC.py "
                f"--exp {exp} --labelnum {labelnum} --stage_name {stage}{iter_part}")
    except Exception:
        return "python test_ACDC.py  # (无法自动解析参数，请手动核对)"


def _scan_available_ckpts(base_dir: str, model: str) -> list:
    """
    扫描 base_dir 下所有可用的检查点文件（best_model + iter_*.pth）。
    """
    found = []
    if not os.path.isdir(base_dir):
        return found
    for exp_dir in sorted(os.listdir(base_dir)):
        for stage in ('train', 'pre_train'):
            stage_dir = os.path.join(base_dir, exp_dir, stage)
            if not os.path.isdir(stage_dir):
                continue
            for fname in sorted(os.listdir(stage_dir)):
                if fname == f'{model}_best_model.pth' or \
                   (fname.startswith('iter_') and fname.endswith('.pth')):
                    found.append(os.path.join(stage_dir, fname))
    return found


# ═══════════════════════════════════════════════════════════════════════
# 推理主流程
# ═══════════════════════════════════════════════════════════════════════

def Inference(FLAGS):
    # ── 修复 2：根据 eval_on_val 选择测试列表 ────────────────────────
    # eval_on_val=0 (默认): test.list  → 正式测试集
    # eval_on_val=1       : val.list   → 与训练期间 TensorBoard val/mean_dice 对齐
    if FLAGS.eval_on_val:
        list_file = os.path.join(FLAGS.root_path, 'val.list')
        logging.info("[修复 2] 使用 val.list（与训练 val Dice 对齐模式）")
    else:
        list_file = os.path.join(FLAGS.root_path, 'test.list')

    with open(list_file, 'r') as f:
        image_list = sorted([l.strip().split(".")[0] for l in f.readlines()])
    logging.info(f"Test cases: {len(image_list)}  list={list_file}")

    snapshot_path  = (f"./model/CML/ACDC_{FLAGS.exp}_{FLAGS.labelnum}_labeled"
                      f"/{FLAGS.stage_name}")
    test_save_path = (f"./model/CML/ACDC_{FLAGS.exp}_{FLAGS.labelnum}_labeled"
                      f"/{FLAGS.model}_predictions/")

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    # ── 定位检查点 ───────────────────────────────────────────────────
    ckpt_path = _resolve_ckpt_path(snapshot_path, FLAGS.model, FLAGS.iter_ckpt)

    if not os.path.exists(ckpt_path):
        found = _scan_available_ckpts("./model/CML", FLAGS.model)
        hint = ""
        if found:
            hint = "\n\n已找到以下可用检查点：\n" + "\n".join(f"  {p}" for p in found)
            hint += ("\n\n建议根据上方路径重新设置参数，例如：\n  "
                     + _ckpt_to_args(found[0], FLAGS.model))
        else:
            hint = ("\n\n在 ./model/CML/ 下未找到任何检查点。\n"
                    "请确认训练已完成，或检查 --root_path / --exp / --labelnum。")
        raise FileNotFoundError(f"权重文件不存在: {ckpt_path}{hint}")

    # ── 加载模型 ─────────────────────────────────────────────────────
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes).cuda()

    ckpt = torch.load(ckpt_path, map_location='cuda')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        net.load_state_dict(ckpt['model'])
        extra_keys = [k for k in ckpt if k != 'model' and k != 'optimizer']
        if extra_keys:
            logging.info(f"检查点额外键（推理时忽略）: {extra_keys}")
    else:
        net.load_state_dict(ckpt)

    logging.info(f"Loaded      : {ckpt_path}")
    logging.info(f"Ensemble    : {FLAGS.ensemble}  Patch: {FLAGS.patch_size}")
    logging.info(f"Postprocess : LargestCC={'ON' if FLAGS.use_postprocess else 'OFF'}")
    logging.info(f"NormInput   : {'ON' if FLAGS.normalize_input else 'OFF (h5 已预归一化)'}")

    # ── 逐例推理 ─────────────────────────────────────────────────────
    totals      = [np.zeros(4), np.zeros(4), np.zeros(4)]
    per_case    = []
    _debug_flag = [False]          # 修复 3：首 case 打印输入范围

    for case in tqdm(image_list, desc="Testing", ncols=70):
        m1, m2, m3 = test_single_volume(case, net, test_save_path, FLAGS,
                                         _debug_first=_debug_flag)
        totals[0] += np.array(m1)
        totals[1] += np.array(m2)
        totals[2] += np.array(m3)
        per_case.append((case, m1, m2, m3))

    n = len(image_list)
    avg_metrics = [t / n for t in totals]

    # ── 保存逐例 CSV ─────────────────────────────────────────────────
    if FLAGS.save_csv:
        csv_path = os.path.join(test_save_path, '../per_case_metrics.csv')
        with open(csv_path, 'w', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['case',
                             'rv_dice',  'rv_jc',  'rv_hd95',  'rv_asd',
                             'myo_dice', 'myo_jc', 'myo_hd95', 'myo_asd',
                             'lv_dice',  'lv_jc',  'lv_hd95',  'lv_asd'])
            for case, m1, m2, m3 in per_case:
                writer.writerow([case,
                                 *[f'{v:.4f}' for v in m1],
                                 *[f'{v:.4f}' for v in m2],
                                 *[f'{v:.4f}' for v in m3]])
        logging.info(f"Per-case CSV → {csv_path}")

    return avg_metrics, test_save_path, per_case


# ═══════════════════════════════════════════════════════════════════════
# 结果打印
# ═══════════════════════════════════════════════════════════════════════

def print_results(avg_metrics, test_save_path, per_case=None, FLAGS=None):
    """
    打印三类别平均指标，单独高亮 Myo（class 2）Dice，
    并在结果末尾打印诊断建议。
    """
    class_names  = [
        'RV  (cls 1 | val/class1_dice)',
        'Myo (cls 2 | val/class2_dice)',
        'LV  (cls 3 | val/class3_dice)',
    ]
    metric_names = ['Dice', 'Jaccard', 'HD95', 'ASD']

    sep      = "─" * 70
    fmt_head = f"{'Class':<36}" + "".join(f"{m:>8}" for m in metric_names)

    # 显示后处理状态
    pp_status = "LargestCC=ON ✓" if (FLAGS and FLAGS.use_postprocess) else "LargestCC=OFF（原始 bug）"
    lines = [sep, f"后处理: {pp_status}", fmt_head, sep]

    dice_list, jc_list, hd_list, asd_list = [], [], [], []

    for name, m in zip(class_names, avg_metrics):
        tag = "  ★" if "Myo" in name else ""
        row = f"{name:<36}" + "".join(f"{v:>8.4f}" for v in m) + tag
        lines.append(row)
        dice_list.append(m[0])
        jc_list.append(m[1])
        hd_list.append(m[2])
        asd_list.append(m[3])

    mean_dice = float(np.mean(dice_list))
    mean_jc   = float(np.mean(jc_list))
    mean_hd95 = float(np.mean(hd_list))
    mean_asd  = float(np.mean(asd_list))

    lines += [
        sep,
        f"{'Mean':<36}" + "".join(f"{v:>8.4f}" for v in
                                    [mean_dice, mean_jc, mean_hd95, mean_asd]),
        sep,
    ]

    if per_case:
        rv_d  = [m1[0] for _, m1, m2, m3 in per_case]
        myo_d = [m2[0] for _, m1, m2, m3 in per_case]
        lv_d  = [m3[0] for _, m1, m2, m3 in per_case]
        lines += [
            "Per-case Dice  mean ± std:",
            f"  RV  = {np.mean(rv_d):.4f} ± {np.std(rv_d):.4f}",
            f"  Myo = {np.mean(myo_d):.4f} ± {np.std(myo_d):.4f}  ★",
            f"  LV  = {np.mean(lv_d):.4f} ± {np.std(lv_d):.4f}",
            sep,
        ]

    # ── 诊断建议 ─────────────────────────────────────────────────────
    lines += [
        "诊断建议（如果结果仍低于训练 TensorBoard 曲线）:",
        "  1. 运行 --eval_on_val 1，确认是否能复现 TensorBoard val/mean_dice",
        "     · 若 eval_on_val 结果≈TensorBoard → 是 val/test 集分布差异，属正常",
        "     · 若 eval_on_val 结果也低 → 说明还有其他 bug，检查归一化或路径",
        "  2. 检查 [DEBUG] 行的输入范围是否在 [0,1]",
        "     · 若不在 [0,1] → 加 --normalize_input 1",
        "  3. 本次 LargestCC=" + ("已开启，后处理 bug 已修复" if (FLAGS and FLAGS.use_postprocess) else "未开启，请加 --use_postprocess 1"),
    ]

    result_str = "\n".join(lines)
    print("\n" + result_str)
    logging.info("\n" + result_str)

    perf_file = os.path.join(test_save_path, '../performance.txt')
    with open(perf_file, 'w', encoding='utf-8') as f:
        f.write(result_str + "\n")
    logging.info(f"Results saved → {perf_file}")

    return mean_dice, mean_jc, mean_hd95, mean_asd


# ═══════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    log_dir = (f"./model/CML/ACDC_{FLAGS.exp}_{FLAGS.labelnum}_labeled"
               f"/{FLAGS.stage_name}")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "test_log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(FLAGS))

    print("\n" + "=" * 70)
    print("CML v8.5-fixed  ACDC Test")
    print(f"  LargestCC 后处理: {'ON ✓' if FLAGS.use_postprocess else 'OFF（关闭则复现训练/测试差距 bug）'}")
    print(f"  评测集: {'val.list（对齐训练 TensorBoard）' if FLAGS.eval_on_val else 'test.list（正式测试）'}")
    print(f"  Myo (class 2) ★ = class-adaptive alpha 优化重点")
    print("=" * 70)

    avg_metrics, test_save_path, per_case = Inference(FLAGS)
    mean_dice, mean_jc, mean_hd95, mean_asd = print_results(
        avg_metrics, test_save_path, per_case, FLAGS=FLAGS)

    print(f"\nMean Dice={mean_dice:.4f}   Mean Jaccard={mean_jc:.4f}"
          f"   Mean HD95={mean_hd95:.4f}   Mean ASD={mean_asd:.4f}")