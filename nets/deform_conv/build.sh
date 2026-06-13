#!/usr/bin/env bash
# Build the deformable-conv CUDA extension (used by the GwcNet/AANet backbones).
#
# TORCH_CUDA_ARCH_LIST controls which GPU architectures are compiled in. The
# default below targets Turing(7.5) through Blackwell(12.0, e.g. RTX 50-series).
# If your GPU is missing you will hit "no kernel image is available" at runtime.
# Find your arch with:  python -c "import torch; print(torch.cuda.get_device_capability())"
# then add it, or override from the environment:
#   TORCH_CUDA_ARCH_LIST="8.9" bash build.sh
# A CUDA toolkit new enough for your arch is required (CUDA >= 12.8 for sm_120).
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5 8.0 8.6 8.9 9.0 12.0}"
PYTHON=${PYTHON:-"python"}

if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace
