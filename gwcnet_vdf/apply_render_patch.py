#!/usr/bin/env python3
"""
apply_render_patch.py — turn an official GwcNet checkout into the
disparityrender variant, changing ONLY the disparity head.

Run this AFTER `git clone https://github.com/xy-guo/GwcNet` so the patch
is applied against the real upstream source (we deliberately do not vendor
a copy of GwcNet, to keep the comparison faithful to the official repo).

What it does (and ONLY this):
  1. Copies disparity_head.py (our disparityrender) into models/.
  2. In models/gwcnet.py:
       - imports disparityrender,
       - in GwcNet.__init__ registers self.render + self.density_temperature,
       - replaces every `F.softmax(cost,1); disparity_regression(...)` head
         with `self.render(cost * self.density_temperature)`.
  The baseline (soft-argmin) repo is left untouched — train it as-is.

Idempotent: re-running detects the marker and refuses to double-patch.

Usage:
    python apply_render_patch.py /path/to/GwcNet
    # produces a patched tree; verify the printed diff hunks before training.
"""
import os
import re
import shutil
import sys

MARKER = "# >>> VDF disparityrender patch <<<"


def fail(msg):
    print(f"[patch][ERROR] {msg}")
    sys.exit(1)


def patch_gwcnet(repo):
    gwc = os.path.join(repo, "models", "gwcnet.py")
    if not os.path.isfile(gwc):
        fail(f"not found: {gwc} — did you clone the official GwcNet here?")

    src = open(gwc, "r").read()
    if MARKER in src:
        print("[patch] already patched, skipping gwcnet.py")
        return

    # 1) import our module (after the existing submodule import)
    if "from models.submodule import" in src:
        src = src.replace(
            "from models.submodule import",
            "from models.disparity_head import disparityrender  " + MARKER + "\nfrom models.submodule import",
            1,
        )
    else:
        src = MARKER + "\nfrom models.disparity_head import disparityrender\n" + src

    # 2) register render + temperature in __init__.
    #    Anchor: the line that sets self.maxdisp = maxdisp inside GwcNet.__init__.
    m = re.search(r"(\n(\s+)self\.maxdisp\s*=\s*maxdisp\s*\n)", src)
    if not m:
        fail("could not find `self.maxdisp = maxdisp` in GwcNet.__init__ — "
             "inspect models/gwcnet.py and patch the head manually.")
    indent = m.group(2)
    inject = (
        f"{indent}# {MARKER.strip('# ')}\n"
        f"{indent}self.render = disparityrender(0, self.maxdisp - 1, self.maxdisp)\n"
        f"{indent}self.density_temperature = torch.nn.Parameter(torch.tensor(1.0))\n"
    )
    src = src[:m.end()] + inject + src[m.end():]

    # 3) replace each soft-argmin head:
    #    pattern: cost{N} = torch.squeeze(cost{N}, 1)
    #             pred{N} = F.softmax(cost{N}, dim=1)
    #             pred{N} = disparity_regression(pred{N}, self.maxdisp)
    head_re = re.compile(
        r"(?P<ind>[ \t]*)(?P<cv>\w+)\s*=\s*torch\.squeeze\((?P=cv),\s*1\)\s*\n"
        r"[ \t]*(?P<pd>\w+)\s*=\s*F\.softmax\((?P=cv),\s*dim=1\)\s*\n"
        r"[ \t]*(?P=pd)\s*=\s*disparity_regression\((?P=pd),\s*self\.maxdisp\)"
    )

    def repl(mo):
        ind, cv, pd = mo.group("ind"), mo.group("cv"), mo.group("pd")
        return (f"{ind}{cv} = torch.squeeze({cv}, 1)\n"
                f"{ind}{pd} = self.render({cv} * self.density_temperature).squeeze(1)  {MARKER}")

    src, n = head_re.subn(repl, src)
    if n == 0:
        fail("no soft-argmin heads matched — the upstream head differs from the "
             "expected `squeeze;softmax;disparity_regression` form. Patch manually "
             "using disparity_head.py docstring as a guide.")
    print(f"[patch] replaced {n} disparity head(s) with disparityrender")

    open(gwc, "w").write(src)
    print(f"[patch] wrote {gwc}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    repo = os.path.abspath(sys.argv[1])
    here = os.path.dirname(os.path.abspath(__file__))

    # copy our disparityrender module into the repo's models/
    dst = os.path.join(repo, "models", "disparity_head.py")
    shutil.copyfile(os.path.join(here, "disparity_head.py"), dst)
    print(f"[patch] copied disparity_head.py -> {dst}")

    patch_gwcnet(repo)
    print("\n[patch] DONE. Review the changes:")
    print(f"        cd {repo} && git diff models/gwcnet.py")
    print("        The ONLY semantic change must be the disparity head + 3 added lines.")


if __name__ == "__main__":
    main()
