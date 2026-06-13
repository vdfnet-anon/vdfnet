"""
disparity_head.py — disparityrender module for the GwcNet second-backbone experiment.

This is the SAME module used in the IGEV experiments
(copied verbatim from igev_baseline/core/submodule.py), so that the
GwcNet ablation uses an identical disparity-rendering operator. Do NOT
edit the math here — keeping it byte-for-byte identical to the IGEV
version is what makes the cross-backbone claim ("same module, two
backbones") valid.

Usage in a GwcNet head (replacing soft-argmin):

    OLD (soft-argmin):
        cost = torch.squeeze(cost, 1)           # [B, D, H, W]
        pred = F.softmax(cost, dim=1)
        pred = disparity_regression(pred, self.maxdisp)

    NEW (disparityrender):
        cost = torch.squeeze(cost, 1)           # [B, D, H, W]
        pred = self.render(cost * self.density_temperature)   # [B, 1, H, W]

where in __init__:
        self.render = disparityrender(0, self.maxdisp - 1, self.maxdisp)
        self.density_temperature = nn.Parameter(torch.tensor(1.0))
"""
import torch
import torch.nn as nn


def disparity_regression(x, maxdisp):
    """Standard soft-argmin (kept for the baseline head)."""
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


class disparityrender(nn.Module):
    """Volume rendering for disparity estimation (VDFNet core contribution).
    Treats cost volume as a density field, applies NeRF-style alpha compositing.

    Identical to igev_baseline/core/submodule.py::disparityrender.
    """
    def __init__(self, mindisp=0, maxdisp=192, dpnum=10):
        super(disparityrender, self).__init__()
        self.register_buffer('disp', torch.linspace(maxdisp, mindisp, dpnum).view(1, -1, 1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        alpha = 1. - torch.exp(-x)
        alpha = torch.flip(alpha, [1])
        T = torch.cumprod(
            torch.cat([
                torch.ones((alpha.shape[0], 1, alpha.shape[2], alpha.shape[3]), device=alpha.device),
                1. - alpha + 1e-10
            ], 1), 1)[:, :-1, :, :]
        weights = alpha * T
        out = torch.sum(weights * self.disp, 1, keepdim=True)
        return out
