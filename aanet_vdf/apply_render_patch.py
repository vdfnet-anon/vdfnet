#!/usr/bin/env python3
"""
apply_render_patch.py — turn an official AANet checkout into the
disparityrender variant, changing ONLY the disparity-estimation head.

Run AFTER `git clone https://github.com/haofeixu/aanet` so the patch is
applied against the real upstream source (we deliberately do NOT vendor a
copy of AANet, to keep the comparison faithful and make Table II's
"AANet-VDF" a genuine AANet backbone rather than a relabel).

AANet's head differs from GwcNet/PSMNet (verified against upstream master):
  - nets/estimation.py defines ONE shared module `DisparityEstimation`,
    but AANet invokes it ONCE PER SCALE (3 scales with SceneFlow defaults),
    each call receiving a cost volume with a DIFFERENT disparity-axis
    length D (max_disp, max_disp//2, max_disp//4). So a fixed-length
    self.disp buffer (as in the canonical disparityrender) does NOT work;
    disp_candidates must be built dynamically per call.
  - the soft-argmax forward is:
        cost_volume = cost_volume if self.match_similarity else -cost_volume
        prob_volume = F.softmax(cost_volume, dim=1)
        disp_candidates = torch.arange(0, D)...view(1, D, 1, 1)
        disp = torch.sum(prob_volume * disp_candidates, 1, keepdim=False)  # [B,H,W]
  - the 2 refinement modules (StereoDRNetRefinement) are pure conv (NO
    softmax/argmin) and are NOT heads — they are left untouched.

What this does (and ONLY this):
  1. Copies disparity_head.py into nets/ for provenance (the math in the
     rewritten forward is byte-identical to that module / IGEV's version).
  2. In nets/estimation.py:
       - adds `self.relu` + `self.density_temperature` to
         DisparityEstimation.__init__ (after `self.max_disp = max_disp`),
       - replaces the WHOLE forward() body with a volume-rendering head
         that keeps the sign convention, handles variable D dynamically,
         and returns [B, H, W] (keepdim=False, NO trailing squeeze).
  The baseline (soft-argmin) clone is left untouched — train it as-is.

Idempotent: re-running detects the marker and refuses to double-patch.

Usage:
    python apply_render_patch.py /path/to/aanet
    # then: cd /path/to/aanet && git diff nets/estimation.py  (verify!)
"""
import os
import re
import shutil
import sys

MARKER = "# >>> VDF disparityrender patch <<<"


def fail(msg):
    print(f"[patch][ERROR] {msg}")
    sys.exit(1)


# Replacement forward body. Mirrors igev_baseline/core/submodule.py::disparityrender:
#   relu -> alpha=1-exp(-x) -> flip -> cumprod transmittance -> sum(weights*disp).
# disp_candidates are built DESCENDING (D-1..0) to match the internal flip,
# and dynamically per call so each scale's D is handled. Output [B,H,W].
NEW_FORWARD = '''    def forward(self, cost_volume):
        {MARKER}
        # Volume-density-field rendering head (replaces soft-argmin).
        assert cost_volume.dim() == 4
        # Keep AANet's sign convention: similarity stays, cost is negated,
        # so the post-ReLU values act as non-negative densities.
        cost_volume = cost_volume if self.match_similarity else -cost_volume
        D = cost_volume.size(1)
        x = self.relu(cost_volume * self.density_temperature)        # [B, D, H, W] >= 0
        alpha = 1. - torch.exp(-x)
        alpha = torch.flip(alpha, [1])                               # channel0 -> far disp
        ones = torch.ones((alpha.size(0), 1, alpha.size(2), alpha.size(3)),
                          device=alpha.device, dtype=alpha.dtype)
        T = torch.cumprod(torch.cat([ones, 1. - alpha + 1e-10], 1), 1)[:, :-1]
        weights = alpha * T
        disp_candidates = torch.arange(D - 1, -1, -1, device=cost_volume.device).type_as(cost_volume)
        disp_candidates = disp_candidates.view(1, D, 1, 1)           # DESCENDING, matches flip
        disp = torch.sum(weights * disp_candidates, 1, keepdim=False)  # [B, H, W]
        return disp
'''.replace("{MARKER}", MARKER)


def patch_estimation(path):
    if not os.path.isfile(path):
        fail(f"not found: {path} — did you clone the official aanet here?")
    src = open(path, "r").read()
    if MARKER in src:
        print("[patch] already patched, skipping nets/estimation.py")
        return

    # 1) __init__: add relu + density_temperature after `self.max_disp = max_disp`
    m = re.search(r"(\n(\s+)self\.max_disp\s*=\s*max_disp\s*\n)", src)
    if not m:
        fail("could not find `self.max_disp = max_disp` in DisparityEstimation.__init__ — "
             "inspect nets/estimation.py and patch manually.")
    indent = m.group(2)
    inject = (
        f"{indent}{MARKER}\n"
        f"{indent}self.relu = torch.nn.ReLU()\n"
        f"{indent}self.density_temperature = torch.nn.Parameter(torch.tensor(1.0))\n"
    )
    src = src[:m.end()] + inject + src[m.end():]

    # 2) replace the WHOLE forward(self, cost_volume) method body.
    #    Match from `def forward(self, cost_volume):` to the `return disp` line.
    fwd_re = re.compile(
        r"[ \t]*def\s+forward\(self,\s*cost_volume\):.*?\n[ \t]*return\s+disp\s*\n",
        re.DOTALL,
    )
    src, n = fwd_re.subn(NEW_FORWARD, src)
    if n != 1:
        fail(f"matched {n} DisparityEstimation.forward methods, expected exactly 1. "
             "Upstream form differs — replace the forward body manually using the "
             "NEW_FORWARD template in this script (keep sign convention + dynamic D).")
    open(path, "w").write(src)
    print("[patch] nets/estimation.py: rewrote DisparityEstimation.forward "
          "(soft-argmin -> disparityrender) + injected relu/temperature")

    # sanity: ensure torch is imported (estimation.py imports torch upstream)
    if re.search(r"^\s*import\s+torch\b", src, re.MULTILINE) is None:
        print("[patch][WARN] `import torch` not detected at top of estimation.py — "
              "verify the rewritten forward can resolve torch.*")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    repo = os.path.abspath(sys.argv[1])
    here = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isdir(os.path.join(repo, "nets")):
        fail(f"{repo}/nets not found — clone haofeixu/aanet first.")
    dst = os.path.join(repo, "nets", "disparity_head.py")
    shutil.copyfile(os.path.join(here, "disparity_head.py"), dst)
    print(f"[patch] copied disparity_head.py -> {dst} (provenance)")

    patch_estimation(os.path.join(repo, "nets", "estimation.py"))

    print("\n[patch] DONE. Review the changes:")
    print(f"        cd {repo} && git diff nets/estimation.py")
    print("        The ONLY semantic change must be: 2 __init__ lines + the "
          "forward-body rewrite in DisparityEstimation. nets/refinement.py, "
          "aanet.py, aggregation.py, cost.py must be UNCHANGED.")


if __name__ == "__main__":
    main()
