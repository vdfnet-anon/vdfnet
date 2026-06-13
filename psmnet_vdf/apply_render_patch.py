#!/usr/bin/env python3
"""
apply_render_patch.py — turn an official PSMNet checkout into the
disparityrender variant, changing ONLY the disparity heads.

Run AFTER `git clone https://github.com/JiaRenChang/PSMNet` so the patch
is applied against the real upstream source (we deliberately do NOT vendor
a copy of PSMNet, to keep the comparison faithful to the official repo and
to make Table II's "PSMNet-VDF" a genuine PSMNet backbone, not a relabel).

PSMNet's head differs from GwcNet (verified against upstream master):
  - models/stackhourglass.py has THREE soft-argmin heads (cost1/2/3),
    each of the form:
        costN = torch.squeeze(costN, 1)
        predN = F.softmax(costN, dim=1)
        predN = disparityregression(self.maxdisp)(predN)
    (preceded by F.upsample(..., mode='trilinear')).
  - it uses a CALLABLE module disparityregression(maxdisp)(pred), not a
    function — so the GwcNet regex will NOT match here.
  - models/basic.py (--model basic) has ONE such head.

What this does (and ONLY this):
  1. Copies disparity_head.py (our disparityrender) into models/.
  2. In models/stackhourglass.py:
       - imports disparityrender,
       - registers self.render + self.density_temperature after
         `self.maxdisp = maxdisp`,
       - replaces all 3 `squeeze;softmax;disparityregression` heads with
         `self.render(costN * self.density_temperature).squeeze(1)`.
       Asserts exactly 3 heads matched (fails loudly otherwise).
  3. (optional) same for models/basic.py if --basic given (asserts 1).
  The baseline (soft-argmin) clone is left untouched — train it as-is.

Idempotent: re-running detects the marker and refuses to double-patch.

Usage:
    python apply_render_patch.py /path/to/PSMNet            # patch stackhourglass
    python apply_render_patch.py /path/to/PSMNet --basic    # also patch basic.py
    # then: cd /path/to/PSMNet && git diff models/   (verify before training)
"""
import os
import re
import shutil
import sys

MARKER = "# >>> VDF disparityrender patch <<<"


def fail(msg):
    print(f"[patch][ERROR] {msg}")
    sys.exit(1)


def _inject_import(src):
    """Import disparityrender after the submodule import (or at top)."""
    if "from models.submodule import" in src:
        return src.replace(
            "from models.submodule import",
            "from models.disparity_head import disparityrender  " + MARKER + "\nfrom models.submodule import",
            1,
        )
    if "from .submodule import" in src:
        return src.replace(
            "from .submodule import",
            "from models.disparity_head import disparityrender  " + MARKER + "\nfrom .submodule import",
            1,
        )
    return MARKER + "\nfrom models.disparity_head import disparityrender\n" + src


def _inject_init(src, classname):
    """Register render + temperature right after `self.maxdisp = maxdisp`."""
    m = re.search(r"(\n(\s+)self\.maxdisp\s*=\s*maxdisp\s*\n)", src)
    if not m:
        fail(f"could not find `self.maxdisp = maxdisp` in {classname}.__init__ — "
             "inspect the file and patch manually.")
    indent = m.group(2)
    inject = (
        f"{indent}{MARKER}\n"
        f"{indent}self.render = disparityrender(0, self.maxdisp - 1, self.maxdisp)\n"
        f"{indent}self.density_temperature = torch.nn.Parameter(torch.tensor(1.0))\n"
    )
    return src[:m.end()] + inject + src[m.end():]


# PSMNet head: costN = torch.squeeze(costN,1)
#              predN = F.softmax(costN, dim=1)
#              [optional comment / blank lines — the upstream cost3 head has
#               two `#For your information...` comments here]
#              predN = disparityregression(self.maxdisp)(predN)
# (whitespace-tolerant; cv/pd captured so we keep the squeeze line and swap the
#  rest. The middle group allows interleaved comment/blank lines so all 3 of
#  PSMNet's heads match — cost3 has comments between softmax and the regression.)
HEAD_RE = re.compile(
    r"(?P<ind>[ \t]*)(?P<cv>\w+)\s*=\s*torch\.squeeze\((?P=cv),\s*1\)\s*\n"
    r"[ \t]*(?P<pd>\w+)\s*=\s*F\.softmax\(\s*(?P=cv)\s*,\s*dim\s*=\s*1\s*\)\s*\n"
    r"(?:[ \t]*#[^\n]*\n|[ \t]*\n)*"
    r"[ \t]*(?P=pd)\s*=\s*disparityregression\(\s*self\.maxdisp\s*\)\(\s*(?P=pd)\s*\)"
)


def _repl(mo):
    ind, cv, pd = mo.group("ind"), mo.group("cv"), mo.group("pd")
    return (f"{ind}{cv} = torch.squeeze({cv}, 1)\n"
            f"{ind}{pd} = self.render({cv} * self.density_temperature).squeeze(1)  {MARKER}")


def patch_file(path, classname, expected_heads):
    if not os.path.isfile(path):
        fail(f"not found: {path} — did you clone the official PSMNet here?")
    src = open(src_path := path, "r").read()
    if MARKER in src:
        print(f"[patch] already patched, skipping {os.path.basename(path)}")
        return
    src = _inject_import(src)
    src = _inject_init(src, classname)
    src, n = HEAD_RE.subn(_repl, src)
    if n != expected_heads:
        fail(f"{os.path.basename(path)}: matched {n} soft-argmin head(s), "
             f"expected exactly {expected_heads}. Upstream head form differs — "
             "patch manually using disparity_head.py docstring as a guide "
             "(replace each `squeeze;softmax;disparityregression(maxdisp)(pred)`).")
    open(src_path, "w").write(src)
    print(f"[patch] {os.path.basename(path)}: replaced {n} head(s) with disparityrender -> wrote")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_basic = "--basic" in sys.argv
    if len(args) != 1:
        print(__doc__)
        sys.exit(1)
    repo = os.path.abspath(args[0])
    here = os.path.dirname(os.path.abspath(__file__))

    dst = os.path.join(repo, "models", "disparity_head.py")
    if not os.path.isdir(os.path.join(repo, "models")):
        fail(f"{repo}/models not found — clone JiaRenChang/PSMNet first.")
    shutil.copyfile(os.path.join(here, "disparity_head.py"), dst)
    print(f"[patch] copied disparity_head.py -> {dst}")

    patch_file(os.path.join(repo, "models", "stackhourglass.py"), "PSMNet", 3)
    if do_basic:
        patch_file(os.path.join(repo, "models", "basic.py"), "PSMNet", 1)

    print("\n[patch] DONE. Review the changes:")
    print(f"        cd {repo} && git diff models/")
    print("        The ONLY semantic change must be: 1 import + 2 __init__ lines "
          "+ the disparity-head swaps (3 in stackhourglass, 1 in basic if --basic).")


if __name__ == "__main__":
    main()
