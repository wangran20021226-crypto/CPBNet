# CPBNet: Collaborative Prototype and Boundary Enhancement Network

Official PyTorch implementation of **Collaborative Prototype and Boundary Enhancement Network for Semi-Supervised Medical Image Segmentation**.

CPBNet is a dual-student semi-supervised medical image segmentation framework. It extends unlabeled supervision from the output space to the **class prototype space** and the **decoder boundary feature space** through:

- **CPAM**: Collaborative Prototype-Aware Module for reliable pixel selection, shared class-prototype construction, and momentum prototype memory.
- **DBM**: Decoder Boundary Module for boundary-aware decoder feature contrast and boundary tangential-normal anisotropy.

## Framework

<p align="center">
  <img src="assets/framework.png" width="95%">
</p>

## Repository structure

```text
CPBNet/
├── CML_ACDC_train.py                  # Training script for ACDC
├── CML_LA_train.py                    # Training script for LA
├── test_ACDC.py                       # Evaluation script for ACDC
├── test_LA.py                         # Evaluation script for LA
├── run_dbm_dice_3_aggressive.sh       # Example ACDC training script
├── decoder_boundary_module.py         # DBM implementation
├── prototype_contrastive_improved.py  # CPAM/prototype learning implementation
├── dataloaders/                       # Dataset loaders and preprocessing scripts
├── networks/                          # Network definitions
└── utils/                             # Training and evaluation utilities
```

## Environment

The code was developed with PyTorch. A typical environment can be created as follows:

```bash
conda create -n cpbnet python=3.8 -y
conda activate cpbnet

# Install PyTorch according to your CUDA version from https://pytorch.org/
pip install torch torchvision torchaudio

pip install numpy scipy scikit-image h5py nibabel SimpleITK medpy tqdm tensorboardX
```

> Please adjust the PyTorch/CUDA version according to your GPU driver.

## Datasets

We evaluate CPBNet on three public medical image segmentation datasets.

| Dataset | Task | Link |
|---|---|---|
| ACDC | 2D cardiac multi-structure segmentation | https://www.creatis.insa-lyon.fr/Challenge/acdc/ |
| LA | 3D left atrium segmentation | https://www.cardiacatlas.org/atriaseg2018-challenge/ |
| BraTS2019 | 3D brain tumor segmentation | https://www.med.upenn.edu/cbica/brats2019/data.html |

After preprocessing, organize the datasets as follows:

```text
data/
├── ACDC/
│   ├── train_slices.list
│   ├── val.list
│   ├── test.list
│   └── data/
│       ├── slices/
│       │   └── *.h5
│       └── *.h5
└── LA/
    ├── train.list
    ├── test.list
    └── 2018LA_Seg_Training Set/
        └── <case_name>/
            └── mri_norm2.h5
```

The default code paths are `../data/ACDC` and `../data/LA`. You can also pass dataset paths by `--root_path`.

## Training

### ACDC

For ACDC, `labelnum=3` corresponds to the 5% labeled setting and `labelnum=7` corresponds to the 10% labeled setting.

**10% labeled setting:**

```bash
bash run_dbm_dice_3_aggressive.sh
```

For LA, `labelnum=4` corresponds to the 5% labeled setting and `labelnum=8` corresponds to the 10% labeled setting.

**10% labeled setting:**

```bash
python CML_LA_train.py \
  --root_path ../data/LA \
  --gpu 0 \
  --labelnum 8 \
  --batch_size 8 \
  --labeled_bs 4 \
  --pre_max_iteration 2000 \
  --train_max_iteration 15000 \
  --base_lr 0.01 \
  --exp LA_10percent
```

**5% labeled setting:**

```bash
python CML_LA_train.py \
  --root_path ../data/LA \
  --gpu 0 \
  --labelnum 4 \
  --batch_size 8 \
  --labeled_bs 4 \
  --pre_max_iteration 2000 \
  --train_max_iteration 15000 \
  --base_lr 0.01 \
  --exp LA_5percent
```

## Testing

### ACDC

Make sure `--exp`, `--labelnum`, and `--stage_name` match the training run. The script loads:

```text
./model/CML/ACDC_<exp>_<labelnum>_labeled/<stage_name>/unet_best_model.pth
```

Example:

```bash
python test_ACDC.py \
  --root_path ../data/ACDC \
  --model unet \
  --num_classes 4 \
  --labelnum 7 \
  --stage_name train \
  --exp Run8E_DBM_ClassAlpha_LowThresh
```

### LA

The LA test script loads:

```text
./model/CML/LA_<exp>_<labelnum>_labeled/<stage_name>/VNet_best_model.pth
```

Example:

```bash
python test_LA.py \
  --root_path ../data/LA \
  --gpu 0 \
  --model VNet \
  --labelnum 8 \
  --stage_name train \
  --exp LA_10percent
```

## Notes

- Training consists of a pre-training stage followed by a self-training stage.
- For ACDC, the default network is `unet` with input size `256 x 256`.
- For LA, the default network is `VNet` with patch size `112 x 112 x 80`.
- Current public scripts provide training/testing examples for ACDC and LA. BraTS2019 is included as a dataset used in the paper; please prepare BraTS2019 according to the official challenge format before adding or running BraTS-specific scripts.

## Citation

If this work is useful for your research, please cite:

```bibtex
@article{wang2026cpbnet,
  title={Collaborative Prototype and Boundary Enhancement Network for Semi-Supervised Medical Image Segmentation},
  author={Wang, Cheng and Wang, Ran and Liu, Xu and Fang, Qiqi and Jiang, Xiaogao and Luo, Qi and Zhu, Jiawen and Li, Wanggen},
  journal={IEEE Transactions on Medical Imaging},
  year={2026}
}
```

## Acknowledgement

This repository is built for semi-supervised medical image segmentation research. We thank the organizers of the ACDC, 2018 Atrial Segmentation Challenge, and BraTS2019 datasets.
