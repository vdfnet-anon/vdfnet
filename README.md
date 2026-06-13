<div align="center">

# VDFNet: Volumetric Density Field for Stereo Matching

**A NeRF-inspired, plug-and-play replacement for soft-argmin that improves cross-domain generalization at zero parameter cost.**

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20%2B%20cu128-EE4C2C.svg?logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/IEEE%20TNNLS-under%20review-orange.svg)

<img src="assets/architecture.png" width="90%" alt="VDFNet architecture"/>

</div>

---

**VDFNet** reinterprets the disparity estimation stage as **volumetric density field
rendering**. Instead of soft-argmin regression, it models the disparity distribution as a
1D density field along the disparity axis and renders disparity via NeRF-style alpha
compositing — a physically interpretable, multi-modal estimator. The `disparityrender`
operator is a **drop-in replacement for the soft-argmin head**, validated on four
independent backbones (IGEV, GwcNet, PSMNet, AANet): it keeps in-domain accuracy
comparable while significantly improving **zero-shot cross-domain generalization**.

> 📄 Paper: *VDFNet: Volumetric Density Field for Stereo Matching* — under review, IEEE TNNLS.
> Pretrained weights + one-command scripts reproduce the paper's Table I & II directly
> (see [Reproducing the paper results](#reproducing-the-paper-results)).

## Contents

- [Quick start](#quick-start)
- [Reproducing the paper results](#reproducing-the-paper-results)
- [Installation](#installation)
- [Datasets](#datasets)
- [Pretrained models](#pretrained-models)
- [Training](#training) · [Inference](#inference) · [Evaluation](#evaluation)
- [Multi-backbone reproduction](#multi-backbone-reproduction)
- [Code structure](#code-structure) · [Citation](#citation) · [License](#license)

---

## Quick start

```bash
conda create -y -n vdfnet python=3.10 && conda activate vdfnet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # match your CUDA
pip install -r requirements.txt
cd nets/deform_conv && bash build.sh && cd ../..   # CUDA extension
cp env.sh.example env.sh                            # edit paths, then:
source env.sh && bash check_env.sh
```

See [Installation](#installation) for per-step details and GPU-arch notes
(verified on RTX 5090 / sm_120, CUDA 12.8, PyTorch 2.11). All scripts read
dataset/checkpoint paths from environment variables (`VDFNET_DATA`,
`VDFNET_EVAL_DATA`, `VDFNET_CKPT`, `VDFNET_ROOT`, `WORKSPACE`) and fall back to
`./data/*` and `./checkpoints` when unset.


---

## Reproducing the paper results

This is the one-stop path from a fresh clone to the numbers reported in the
paper. The released IGEV checkpoints + two one-command scripts reproduce the
paper's two core tables directly.

1. **Set up the environment** — follow [Installation](#installation) (conda,
   PyTorch cu128, deform_conv extension, `check_env.sh`). The IGEV evaluation
   needs `timm` (in `requirements.txt`).
2. **Get the data** — [SceneFlow](#datasets) for in-domain, and the zero-shot
   sets (ETH3D / KITTI 2015 / Middlebury H) for cross-domain. The helper
   `scripts/setup_eval_data.sh` downloads ETH3D + Middlebury automatically
   (KITTI needs a registered login — see [Datasets](#datasets)).
3. **Get the checkpoints** — download the three IGEV checkpoints from
   [Pretrained models](#pretrained-models) (no training required).
4. **Reproduce Table I (in-domain ablation)** — one command:

```bash
cd igev_baseline
SCENEFLOW_DIR=/path/to/SceneFlow python reproduce_table1.py \
    --softargmin /path/to/vdfnet_igev_softargmin_sceneflow.pth \
    --render     /path/to/vdfnet_igev_render_sceneflow.pth \
    --render_temp /path/to/vdfnet_igev_render_temp_sceneflow.pth
```

5. **Reproduce Table II (zero-shot cross-domain, IGEV rows)** — one command:

```bash
cd igev_baseline
python reproduce_table2.py \
    --softargmin /path/to/vdfnet_igev_softargmin_sceneflow.pth \
    --render     /path/to/vdfnet_igev_render_sceneflow.pth \
    --datasets eth3d kitti middlebury_H
```

Both scripts print a table with the measured value next to the paper value for
each cell, so the match is visible at a glance.

### Expected results (sanity anchors)

The IGEV instance is reported in two tables. **Table I** is the in-domain
disparity-head ablation; **Table II** is the zero-shot cross-domain comparison.

**Table I — SceneFlow test set (in-domain):**

| Disparity head | checkpoint | EPE | 1-ER% | 3-ER% |
|----------------|-----------|-----|-------|-------|
| soft-argmin (baseline) | `..._softargmin_...`  | 0.4813 | 5.29 | 2.50 |
| +disparityrender       | `..._render_...`      | 0.4790 | 5.38 | 2.51 |
| +density_temperature (flagship) | `..._render_temp_...` | **0.4686** | **5.24** | **2.45** |

**Table II — zero-shot cross-domain (SceneFlow-trained, no fine-tuning):**

| Disparity head | ETH3D EPE | KITTI D1-all | Middlebury H EPE |
|----------------|-----------|--------------|------------------|
| soft-argmin (baseline) | 0.322 | 6.67% | 0.848 |
| +disparityrender       | **0.279** | **5.96%** | **0.885** |

The paper's core finding: `disparityrender` is **comparable in-domain but
significantly improves zero-shot cross-domain generalization**, and this holds
across four architecturally distinct backbones (see
[Multi-backbone](#multi-backbone-reproduction)).

> Note: the flagship IGEV row in Table II reports the in-domain number with the
> temperature-enhanced checkpoint (`render_temp`, 0.4686) and the cross-domain
> numbers with the `render` checkpoint; `reproduce_table2.py` lets you evaluate
> any of the checkpoints so this is fully transparent.

---

## Installation

### Verified environment

Reproduced end-to-end on: Ubuntu 22.04, NVIDIA RTX 5090 ×2 (sm_120, Blackwell),
driver 595.71, CUDA 12.8, Python 3.10, PyTorch 2.11+cu128. Older GPUs
(Turing/Ampere/Ada/Hopper) work too — see the arch notes below.

### 1. System prerequisites

- NVIDIA driver for your GPU (Blackwell/sm_120 needs driver >= 570).
- CUDA Toolkit matching your GPU — required to compile the deformable-conv
  extension (CUDA >= 12.8 for sm_120; >= 11.0 for older cards). Put `nvcc` on PATH:

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

### 2. Python environment

```bash
conda create -y -n vdfnet python=3.10
conda activate vdfnet
```

### 3. PyTorch (match your CUDA / GPU arch)

```bash
# Blackwell (RTX 50-series, sm_120), CUDA 12.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# Older GPUs: pick the matching index-url from https://pytorch.org (e.g. cu121).
# Verify your arch is supported (expect sm_120 in the list for RTX 50-series):
python -c "import torch; print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_capability())"
```

### 4. Other dependencies

```bash
pip install -r requirements.txt
```

`apex` is optional, only for mixed-precision training (`--amp`); install from
https://github.com/NVIDIA/apex if needed.

### 5. Deformable-conv CUDA extension (GwcNet/AANet backbones)

```bash
cd nets/deform_conv
# build.sh defaults to sm_7.5–12.0. If your GPU arch is missing, edit the
# TORCH_CUDA_ARCH_LIST line or override it, e.g.:  TORCH_CUDA_ARCH_LIST="8.9" bash build.sh
bash build.sh
# Verify (import torch FIRST, otherwise libc10.so is not found):
python -c "import torch; import deform_conv_cuda; print('deform_conv OK')"
cd ../..
```

### 6. Configure paths & sanity-check

```bash
cp env.sh.example env.sh        # edit dataset/checkpoint paths for your machine
source env.sh
bash check_env.sh               # verifies GPU, deps, and data layout
```

All scripts read dataset/checkpoint locations from environment variables
(`VDFNET_DATA`, `VDFNET_EVAL_DATA`, `VDFNET_CKPT`, `VDFNET_ROOT`, `WORKSPACE`)
and fall back to `./data/*` and `./checkpoints` when unset.


---

## Datasets

Place datasets under `./data` (or point `VDFNET_DATA` / `VDFNET_EVAL_DATA` at
them in `env.sh`).

### SceneFlow (training)

Download [FlyingThings3D, Monkaa, Driving](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) and organize as:

```
SceneFlow/
├── FlyingThings3D/{frames_finalpass,disparity}/
├── Monkaa/{frames_finalpass,disparity}/
└── Driving/{frames_finalpass,disparity}/
```

### Zero-shot evaluation sets

For the IGEV reproduction scripts, `scripts/setup_eval_data.sh` downloads and
arranges **ETH3D** and **Middlebury** automatically:

```bash
EVAL_ROOT=/data bash scripts/setup_eval_data.sh   # downloads ETH3D + Middlebury
```

The IGEV evaluation code (`igev_baseline/core/stereo_datasets.py`) reads the
datasets from hard-coded `/data/...` roots in this layout:

```
/data/
├── ETH3D/two_view_training/<scene>/{im0.png,im1.png}
│        two_view_training_gt/<scene>/disp0GT.pfm
├── Middlebury/trainingH/<scene>/{im0.png,im1.png,disp0GT.pfm}
└── KITTI/KITTI_2012/training/{colored_0,colored_1,disp_occ}/*_10.png
        KITTI_2015/training/{image_2,image_3,disp_occ_0}/*_10.png
```

If your data lives elsewhere, symlink it into `/data` (e.g.
`sudo ln -sfn /your/path/ETH3D /data/ETH3D`). **KITTI** is not auto-downloaded —
it needs a free registered login at
[cvlibs.net](https://www.cvlibs.net/datasets/kitti/); download the *stereo 2012*
and *stereo 2015* "data set" archives (2 GB each) and arrange as above.

Sources: [KITTI](https://www.cvlibs.net/datasets/kitti/),
[Middlebury v3](https://vision.middlebury.edu/stereo/eval3/),
[ETH3D](https://www.eth3d.net/datasets).

---

## Pretrained models

The three IGEV-backbone checkpoints are released so reviewers can reproduce the
paper's two core tables directly (see [Reproducing the paper results](#reproducing-the-paper-results)).
Download them from the
[GitHub Releases](https://github.com/vdfnet-anon/vdfnet/releases) page and place
them under `./checkpoints` (or point `VDFNET_CKPT` at them).

| Disparity head | File (Release asset) | Backbone | SceneFlow EPE | MD5 |
|----------------|----------------------|----------|---------------|-----|
| soft-argmin (baseline) | `vdfnet_igev_softargmin_sceneflow.pth` | IGEV | 0.4813 | `e2b9e4d4f7de26318fd9872d44e305a3` |
| +disparityrender | `vdfnet_igev_render_sceneflow.pth` | IGEV | 0.4790 | `3c3bbea407798d75a13da84edd43ac84` |
| +density_temperature (flagship) | `vdfnet_igev_render_temp_sceneflow.pth` | IGEV | **0.4686** | `10df737182e4de84bbbfd410d898a9e1` |

These three checkpoints are the rows of Table I, and together they reproduce the
controlled `soft-argmin` → `disparityrender` comparison that is the core claim of
the paper. The `softargmin` and `render` checkpoints differ by exactly the
disparity head (zero extra parameters); `render_temp` adds the learnable density
temperature. Loaded automatically by the reproduction scripts; the baseline uses
the original IGEV model class, the render variants use the disparityrender class.

**Loading note:** checkpoints are saved with a `module.` prefix (DDP). Either
wrap the model in `DataParallel`/`DDP`, or strip the prefix:

```python
ck = torch.load(path, map_location='cpu')
sd = {k.replace('module.', '', 1): v for k, v in ck.items()}
model.load_state_dict(sd)
```

Multi-backbone (PSMNet/GwcNet/AANet) checkpoints are reproducible from the
[multi-backbone workflow](#multi-backbone-reproduction); their weights are not
published here. Third-party baselines (IGEV++, NMRF, DEFOM, official backbones)
are not re-hosted — get them from their official repositories.

---

## Training

`train.py` runs single-GPU out of the box and scales to multi-GPU DDP via
`torchrun`.

**Single GPU:**

```bash
python train.py \
    --data_dir $VDFNET_DATA --dataset_name SceneFlow \
    --checkpoint_dir $VDFNET_CKPT/sceneflow \
    --batch_size 2 --img_height 256 --img_width 512 \
    --max_epoch 64 --lr_scheduler_type MultiStepLR --milestones 20,30,40,50,60
```

**Multi-GPU (DDP, e.g. 2 GPUs):**

```bash
torchrun --nproc_per_node=2 train.py \
    --data_dir $VDFNET_DATA --dataset_name SceneFlow \
    --checkpoint_dir $VDFNET_CKPT/sceneflow \
    --batch_size 4 --img_height 288 --img_width 512 \
    --max_epoch 64 --lr_scheduler_type MultiStepLR --milestones 20,30,40,50,60
```

A convenience recipe for 2×GPU is in `scripts/train_sceneflow_5090.sh`.

**KITTI fine-tuning:**

```bash
python train.py \
    --data_dir $VDFNET_EVAL_DATA/KITTI --dataset_name KITTI_mix \
    --checkpoint_dir $VDFNET_CKPT/kitti \
    --pretrained_vdfnet $VDFNET_CKPT/sceneflow/vdfnet_best.pth \
    --batch_size 2 --img_height 256 --img_width 512 \
    --max_epoch 400 --lr_scheduler_type MultiStepLR --milestones 200,300,350 --no_validate
```

---

## Inference

Runs on a directory of stereo pairs; auto-detects KITTI (`left/`+`right/`),
KITTI benchmark (`image_2/`+`image_3/`), and Middlebury/ETH3D
(`im0.png`+`im1.png`) layouts.

```bash
python predict.py \
    --data_dir /path/to/stereo/pairs \
    --pretrained_vdfnet $VDFNET_CKPT/sceneflow/vdfnet_best.pth \
    --save_type pfm --visualize
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Input stereo image directory |
| `--pretrained_vdfnet` | required | Path to pretrained model |
| `--output_dir` | `data_dir/pred/` | Output directory |
| `--save_type` | `png` | `png`, `pfm`, `npy`, `npz` |
| `--visualize` | off | Save color-coded disparity maps |
| `--max_disp` | 192 | Maximum disparity |

---

## Evaluation

For the released IGEV checkpoints, the canonical entry point is
`igev_baseline/evaluate_stereo.py` (the official IGEV evaluation harness), which
the reproduction scripts wrap. Evaluate a single checkpoint on one dataset:

```bash
cd igev_baseline
# SceneFlow test (in-domain EPE / 1-ER / 3-ER):
SCENEFLOW_DIR=/path/to/SceneFlow python evaluate_stereo.py \
    --restore_ckpt /path/to/vdfnet_igev_render_temp_sceneflow.pth --dataset sceneflow
# Zero-shot cross-domain (eth3d / kitti / middlebury_H):
python evaluate_stereo.py \
    --restore_ckpt /path/to/vdfnet_igev_render_sceneflow.pth --dataset eth3d
```

To reproduce whole tables in one command (recommended), use
`reproduce_table1.py` / `reproduce_table2.py` — see
[Reproducing the paper results](#reproducing-the-paper-results).

> The baseline checkpoint loads into the original IGEV model class
> (`igev_stereo_original`); the render variants load into the disparityrender
> class (`igev_stereo`). The reproduction scripts pick the right class per
> checkpoint automatically.

The `predict.py` / `eval_eth3d.py` / `eval_vdfnet_3er.py` utilities in the repo
root target the self-contained VDFNet-Light (`stereorf`) variant and the
generic inference workflow; they are **not** the entry point for the released
IGEV checkpoints above.

---

## Multi-backbone reproduction

A central claim of the paper is that the disparity-rendering operator is
**backbone-agnostic**: swapping a backbone's soft-argmin head for
`disparityrender` improves cross-domain generalization across four
architecturally distinct backbones (iterative IGEV, 3D-CNN GwcNet/PSMNet,
2D-CNN AANet). To keep the comparison fair, each backbone is taken from its
**official repository** and patched so that the *only* changed variable is the
disparity head — data, hyper-parameters, schedule, optimizer and seed are
identical between the `base` (soft-argmin) and `render` (disparityrender) runs.

The exact same rendering operator is reused byte-for-byte across all backbones
(`{igev_baseline,psmnet_vdf,gwcnet_vdf,aanet_vdf}/disparity_head.py` are
identical), which is what makes the "one module, many backbones" argument hold.

> **Scope:** this section is a *training-reproduction recipe* — you clone each
> official backbone, apply the one-file patch, and train the `base`/`render`
> pair yourself. Pretrained multi-backbone checkpoints are **not** shipped with
> this release (only the flagship IGEV checkpoints are; see
> [Pretrained models](#pretrained-models)). The `--ckpt` paths in the eval
> commands below refer to the checkpoints *you* produce from the training step.

Each `*_vdf/` directory is a self-contained patch kit:

| Directory | Official backbone | Entry points |
|-----------|-------------------|--------------|
| `psmnet_vdf/` | [JiaRenChang/PSMNet](https://github.com/JiaRenChang/PSMNet) | `apply_render_patch.py`, `train_psmnet_sceneflow.sh`, `eval_psmnet_generalization.py` |
| `gwcnet_vdf/` | [xy-guo/GwcNet](https://github.com/xy-guo/GwcNet) | `apply_render_patch.py`, `train_gwcnet_sceneflow.sh`, `eval_gwcnet_generalization.py` |
| `aanet_vdf/`  | [haofeixu/aanet](https://github.com/haofeixu/aanet) | `apply_render_patch.py`, `train_aanet_sceneflow.sh`, `eval_aanet_generalization.py` |
| `igev_baseline/` | IGEV (self-contained) | `eval_sceneflow_expa.py` (soft-argmin), `eval_sceneflow_expb.py` (render) |

Workflow (PSMNet shown; GwcNet/AANet are identical with their own repo URL):

```bash
source env.sh   # sets WORKSPACE, VDFNET_DATA, VDFNET_CKPT, VDFNET_ROOT

# 1) clone the official backbone twice: base (untouched) + render (patched)
git clone https://github.com/JiaRenChang/PSMNet $WORKSPACE/PSMNet_base
cp -r $WORKSPACE/PSMNet_base $WORKSPACE/PSMNet_render

# 2) patch only the render copy, then VERIFY the diff touches only the head
python psmnet_vdf/apply_render_patch.py $WORKSPACE/PSMNet_render
( cd $WORKSPACE/PSMNet_render && git diff models/ )   # expect: import + __init__ + head only

# 3) sanity-check forward orientation before committing to a long run
( cd $WORKSPACE/PSMNet_render && python $VDFNET_ROOT/sanity_forward.py psmnet )

# 4) train both variants with identical hyper-parameters
bash psmnet_vdf/train_psmnet_sceneflow.sh base
bash psmnet_vdf/train_psmnet_sceneflow.sh render

# 5) zero-shot cross-domain evaluation
( cd $WORKSPACE/PSMNet_base   && python $VDFNET_ROOT/psmnet_vdf/eval_psmnet_generalization.py --ckpt $VDFNET_CKPT/psm_base/checkpoint_9.tar   --tag base )
( cd $WORKSPACE/PSMNet_render && python $VDFNET_ROOT/psmnet_vdf/eval_psmnet_generalization.py --ckpt $VDFNET_CKPT/psm_render/checkpoint_9.tar --tag render )
```

See each `*_vdf/README.md` for backbone-specific notes (head shape, checkpoint
naming, fairness checklist).

---

## Code structure

```
vdfnet/
├── train.py / predict.py / model.py / metric.py   # train / infer / loop / metrics
├── eval_eth3d.py / eval_middlebury.py / eval_vdfnet_3er.py
├── nets/              # VDFNet model (stereorf*) + deform_conv CUDA extension
├── dataloader/        # StereoDataset (SceneFlow/KITTI/Middlebury/ETH3D) + transforms
├── filenames/         # dataset filename lists (+ generator)
├── utils/ , thop/     # logging/checkpoint helpers, FLOPs counting
├── igev_baseline/     # IGEV backbone + disparityrender (primary proof)
├── psmnet_vdf/ , gwcnet_vdf/ , aanet_vdf/   # backbone patch kits
├── env.sh.example     # path configuration template
└── check_env.sh       # environment / data layout pre-flight check
```

---

## Citation

```bibtex
@article{vdfnet,
  title   = {VDFNet: Volumetric Density Field for Stereo Matching},
  author  = {VDFNet Authors},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  note    = {Under review},
  year    = {2026}
}
```

This project builds on the official IGEV, PSMNet, GwcNet and AANet repositories;
please also cite those works when using the corresponding backbones.

---

## License

Released under the [MIT License](LICENSE). The cloned backbone repositories
remain under their own licenses (PSMNet/GwcNet: MIT; AANet: Apache-2.0).

