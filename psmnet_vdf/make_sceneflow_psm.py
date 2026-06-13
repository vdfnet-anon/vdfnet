#!/usr/bin/env python3
"""
make_sceneflow_psm.py — build a symlink view of SceneFlow that the OFFICIAL
PSMNet dataloader (dataloader/listflowfile.py) can read UNCHANGED.

PSMNet's listflowfile.py expects, directly under the datapath, six flatly
named folders (it picks them by substring match):
    monkaa_frames_cleanpass/<scene>/{left,right}/*.png
    monkaa_disparity/<scene>/left/*.pfm
    frames_cleanpass/{TRAIN,TEST}/{A,B,C}/<seq>/{left,right}/*.png   (FT3D)
    frames_disparity/{TRAIN,TEST}/{A,B,C}/<seq>/left/*.pfm
    driving_frames_cleanpass/<focal>/<dir>/<speed>/{left,right}/*.png
    driving_disparity/<focal>/<dir>/<speed>/left/*.pfm

The SceneFlow_hf tree is split by subset instead:
    SceneFlow_hf/{Monkaa,Driving,FlyingThings3D}/{frames_finalpass,disparity}/...

The internal structure is identical — only the top-level naming differs — so we
create six DIRECTORY symlinks (no data copied), idempotent, then VALIDATE by
calling PSMNet's own dataloader and printing the train/test counts.

NOTE: PSMNet concatenates `filepath + foldername` with NO separator, so the
datapath you pass to main.py MUST end in '/'. The train script handles this;
this validator appends it automatically.

Usage:
  python make_sceneflow_psm.py \
      --src $VDFNET_DATA \
      --dst $VDFNET_DATA/../SceneFlow_psm \
      --psm $WORKSPACE/PSMNet_base        # to validate via PSMNet's own loader
"""
import argparse
import os
import sys

# PSMNet flat name  ->  (SceneFlow_hf subset, modal subfolder)
MAPPING = {
    "monkaa_frames_cleanpass":  ("Monkaa",         "frames_finalpass"),
    "monkaa_disparity":         ("Monkaa",         "disparity"),
    "frames_cleanpass":         ("FlyingThings3D", "frames_finalpass"),
    "frames_disparity":         ("FlyingThings3D", "disparity"),
    "driving_frames_cleanpass": ("Driving",        "frames_finalpass"),
    "driving_disparity":        ("Driving",        "disparity"),
}


def build(src, dst):
    os.makedirs(dst, exist_ok=True)
    rows = []
    for flat, (subset, modal) in MAPPING.items():
        target = os.path.join(src, subset, modal)
        linkpath = os.path.join(dst, flat)
        status = ""
        if not os.path.isdir(target):
            status = "MISSING-SRC"
        elif os.path.islink(linkpath) or os.path.exists(linkpath):
            if os.path.realpath(linkpath) == os.path.realpath(target):
                status = "ok (exists)"
            else:
                status = "CONFLICT (points elsewhere)"
        else:
            os.symlink(target, linkpath)
            status = "linked"
        rows.append((flat, target, status))
    return rows


def validate(dst, psm_repo):
    """Call PSMNet's own dataloader against the view; return counts or error."""
    sys.path.insert(0, psm_repo)
    try:
        from dataloader import listflowfile as lt
    except Exception as e:  # noqa: BLE001
        return f"import-failed: {e}"
    # PSMNet concatenates filepath + foldername with no '/', so trailing slash:
    fp = dst if dst.endswith("/") else dst + "/"
    try:
        tl, tr, td, vl, vr, vd = lt.dataloader(fp)
    except Exception as e:  # noqa: BLE001
        return f"dataloader-raised: {e}"
    return {
        "train_left": len(tl), "train_right": len(tr), "train_disp": len(td),
        "test_left": len(vl), "test_right": len(vr), "test_disp": len(vd),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.environ.get("VDFNET_DATA", "data/SceneFlow"))
    ap.add_argument("--dst", default=os.environ.get("VDFNET_DATA", "data/SceneFlow") + "_psm")
    ap.add_argument("--psm", default=os.path.join(os.environ.get("WORKSPACE", "."), "PSMNet_base"),
                    help="PSMNet repo to validate via its own dataloader")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        sys.exit(f"[ERR] src not found: {args.src}")

    print(f"[psm-view] building symlink view: {args.dst}  <-  {args.src}")
    rows = build(args.src, args.dst)
    bad = False
    for flat, target, status in rows:
        flag = "" if status in ("linked", "ok (exists)") else "   <-- CHECK"
        if flag:
            bad = True
        print(f"   {flat:26s} -> {target:48s} [{status}]{flag}")

    if bad:
        print("\n[WARN] some links did not resolve cleanly — fix before training.")
        return

    if args.psm and os.path.isdir(args.psm):
        print(f"\n[psm-view] validating via PSMNet dataloader ({args.psm}) ...")
        res = validate(args.dst, args.psm)
        if isinstance(res, str):
            print(f"   [FAIL] {res}")
            print("   Paste this back; do NOT start training yet.")
            return
        print(f"   train: left={res['train_left']} right={res['train_right']} "
              f"disp={res['train_disp']}")
        print(f"   test:  left={res['test_left']} right={res['test_right']} "
              f"disp={res['test_disp']}")
        print("-" * 56)
        # SceneFlow finalpass: ~35454 train / ~4370 test (FT3D test only)
        if (res["train_left"] == res["train_right"] == res["train_disp"]
                and res["test_left"] == res["test_right"] == res["test_disp"]
                and res["train_left"] > 30000 and res["test_left"] > 4000):
            print(f"[PASS] PSMNet reads the view. Train with (note trailing /):")
            print(f"       SCENEFLOW_DIR={args.dst}/")
        else:
            print("[WARN] counts look off (left/right/disp mismatch or low). "
                  "Paste this back before training.")
    else:
        print(f"\n[psm-view] (no --psm repo to validate; view built at {args.dst})")


if __name__ == "__main__":
    main()
