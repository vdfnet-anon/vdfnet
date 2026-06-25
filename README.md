<div align="center">

# VDFNet: A NeRF-Inspired Volume Density Field Rendering Paradigm for Stereo Matching

**A plug-and-play replacement for soft-argmin that improves cross-domain generalization at zero parameter cost.**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20%2B%20cu128-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Status](https://img.shields.io/badge/IEEE%20TNNLS-under%20review-orange.svg)]()

<img src="assets/architecture.png" width="90%" alt="VDFNet architecture"/>

</div>

---

**VDFNet** replaces the soft-argmin disparity head with `disparityrender` — a NeRF-style alpha compositing module that models the cost volume as a 1D density field. It is **architecture-agnostic**: validated on four backbones (IGEV, GwcNet, PSMNet, AANet), it maintains in-domain accuracy while significantly improving **zero-shot cross-domain generalization**, at zero extra parameters.

> 📄 *VDFNet: A NeRF-Inspired Volume Density Field Rendering Paradigm for Stereo Matching* — under review, IEEE TNNLS.

> [!IMPORTANT]
> This repository provides evaluation code and pretrained weights for reviewer reproduction.
> **The complete codebase will be open-sourced upon acceptance.**

---

## Results

### Zero-shot cross-domain generalization

Trained on SceneFlow only — **no fine-tuning** on target domains. VDFNet (`disparityrender`) produces sharper boundaries and fewer large errors than RAFT-Stereo and IGEV-Stereo across ETH3D, KITTI 2015, and Middlebury H.

<img src="assets/exp2.png" width="100%" alt="Zero-shot generalization: ETH3D / KITTI 2015 / Middlebury H"/>

### KITTI 2015 benchmark

<img src="assets/kittiVDF.png" width="100%" alt="KITTI 2015 visual results"/>

### Key numbers

| Setting | Metric | soft-argmin | disparityrender | Δ |
|---------|--------|:-----------:|:---------------:|:---:|
| SceneFlow (in-domain) | EPE | 0.4813 | **0.4686** | −2.6% |
| ETH3D (zero-shot) | EPE | 0.322 | **0.279** | −13.2% |
| KITTI 2015 (zero-shot) | D1-all | 6.67% | **5.96%** | −10.6% |
| Middlebury H (zero-shot) | bad-2.0 | 6.31% | **5.75%** | −8.9% |

> Zero parameter overhead · negligible latency · architecture-agnostic

---

## Reproducing the paper results

1. **Environment** — see [Installation](#installation)
2. **Data** — [SceneFlow](#datasets) for in-domain; run `bash scripts/setup_eval_data.sh` for ETH3D + Middlebury
3. **Checkpoints** — download from [Releases](https://github.com/vdfnet-anon/vdfnet/releases)
4. **Table I** (in-domain ablation):

```bash
cd igev_baseline
SCENEFLOW_DIR=/path/to/SceneFlow python reproduce_table1.py \
    --softargmin /path/to/vdfnet_igev_softargmin_sceneflow.pth \
    --render     /path/to/vdfnet_igev_render_sceneflow.pth \
    --render_temp /path/to/vdfnet_igev_render_temp_sceneflow.pth
```

5. **Table II** (zero-shot cross-domain):

```bash
cd igev_baseline
python reproduce_table2.py \
    --softargmin /path/to/vdfnet_igev_softargmin_sceneflow.pth \
    --render     /path/to/vdfnet_igev_render_sceneflow.pth \
    --datasets eth3d kitti middlebury_H
```

Both scripts print measured vs. paper values side by side.

---

## Installation

```bash
conda create -y -n vdfnet python=3.10 && conda activate vdfnet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
cp env.sh.example env.sh   # edit VDFNET_DATA, VDFNET_EVAL_DATA, VDFNET_CKPT
source env.sh && bash check_env.sh
```

Verified on Ubuntu 22.04, RTX 5090 ×2 (sm_120), CUDA 12.8, PyTorch 2.11. Older GPUs work with a matching PyTorch index URL.

---

## Datasets

**SceneFlow** — download [FlyingThings3D, Monkaa, Driving](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html):

```
SceneFlow/
├── FlyingThings3D/{frames_finalpass,disparity}/
├── Monkaa/{frames_finalpass,disparity}/
└── Driving/{frames_finalpass,disparity}/
```

**ETH3D + Middlebury** — auto-downloaded:
```bash
EVAL_ROOT=/data bash scripts/setup_eval_data.sh
```

**KITTI** — requires free registration at [cvlibs.net](https://www.cvlibs.net/datasets/kitti/). Place under `/data/KITTI/`.

---

## Pretrained models

Download from the [Releases](https://github.com/vdfnet-anon/vdfnet/releases) page and place under `./checkpoints`.

| Checkpoint | Description | SceneFlow EPE | MD5 |
|------------|-------------|:-------------:|-----|
| `vdfnet_igev_softargmin_sceneflow.pth` | soft-argmin baseline | 0.4813 | `e2b9e4d4` |
| `vdfnet_igev_render_sceneflow.pth` | +disparityrender | 0.4790 | `3c3bbea4` |
| `vdfnet_igev_render_temp_sceneflow.pth` | +density\_temperature ★ | **0.4686** | `10df7371` |

> ★ Flagship checkpoint. Use `render` for cross-domain evaluation, `render_temp` for in-domain.

**Loading note** — checkpoints use a `module.` prefix (DDP). Strip it if needed:
```python
ck = torch.load(path, map_location='cpu')
sd = {k.replace('module.', '', 1): v for k, v in ck.items()}
model.load_state_dict(sd)
```

---

## Evaluation

```bash
cd igev_baseline

# In-domain (SceneFlow):
SCENEFLOW_DIR=/path/to/SceneFlow python evaluate_stereo.py \
    --restore_ckpt /path/to/vdfnet_igev_render_temp_sceneflow.pth --dataset sceneflow

# Zero-shot cross-domain:
python evaluate_stereo.py --restore_ckpt /path/to/vdfnet_igev_render_sceneflow.pth --dataset eth3d
python evaluate_stereo.py --restore_ckpt /path/to/vdfnet_igev_render_sceneflow.pth --dataset kitti
python evaluate_stereo.py --restore_ckpt /path/to/vdfnet_igev_render_sceneflow.pth --dataset middlebury_H
```

---

## Citation

```bibtex
@article{vdfnet,
  title   = {VDFNet: A NeRF-Inspired Volume Density Field Rendering Paradigm for Stereo Matching},
  author  = {VDFNet Authors},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  note    = {Under review},
  year    = {2026}
}
```

---

## License

Released under the [MIT License](LICENSE).
