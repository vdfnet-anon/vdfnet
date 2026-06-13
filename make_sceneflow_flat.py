#!/usr/bin/env python3
"""
make_sceneflow_flat.py — build a FLAT symlink view of SceneFlow so the
official GwcNet / PSMNet dataloaders (which expect
  frames_finalpass/TRAIN/<...>, frames_finalpass/TEST/<...>,
  disparity/TRAIN/<...>,        disparity/TEST/<...>)
work against an SceneFlow_hf tree that is split by subset:
  SceneFlow_hf/{Driving,FlyingThings3D,Monkaa}/{frames_finalpass,disparity}/...

It creates ONLY symlinks (no data copied), is idempotent (safe to re-run),
and then VALIDATES by resolving a sample of paths from GwcNet's
filenames/sceneflow_{train,test}.txt.

The mapping mirrors how the official SceneFlow lists are built:
  TRAIN  <-  FlyingThings3D/.../TRAIN/*   (A/B/C subdirs)
             Monkaa/.../<scene>           (all monkaa scenes -> train)
             Driving/.../<focallength>/.. (all driving -> train)
  TEST   <-  FlyingThings3D/.../TEST/*

Usage:
  python make_sceneflow_flat.py \
      --src $VDFNET_DATA \
      --dst $VDFNET_DATA/../SceneFlow_flat \
      --gwc $WORKSPACE/GwcNet_base        # optional: validate against its lists
"""
import argparse
import os
import sys


def link(src, dst):
    """Create a symlink dst -> src, idempotently."""
    if os.path.islink(dst) or os.path.exists(dst):
        return  # already there
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)


def link_children(src_dir, dst_dir):
    """Symlink each immediate child of src_dir into dst_dir.
    Returns (#linked, #skipped_conflict)."""
    linked = conflict = 0
    if not os.path.isdir(src_dir):
        return 0, 0
    for name in os.listdir(src_dir):
        s = os.path.join(src_dir, name)
        d = os.path.join(dst_dir, name)
        if os.path.islink(d) or os.path.exists(d):
            # name clash across subsets — keep first, report it
            if os.path.realpath(d) != os.path.realpath(s):
                conflict += 1
            continue
        os.makedirs(dst_dir, exist_ok=True)
        os.symlink(s, d)
        linked += 1
    return linked, conflict


def build(src, dst):
    """Build the flat view. Returns a log dict."""
    log = {}
    for modal in ("frames_finalpass", "disparity"):
        for split in ("TRAIN", "TEST"):
            os.makedirs(os.path.join(dst, modal, split), exist_ok=True)

        # FlyingThings3D already has TRAIN/TEST — link their children through.
        ft = os.path.join(src, "FlyingThings3D", modal)
        for split in ("TRAIN", "TEST"):
            l, c = link_children(os.path.join(ft, split),
                                 os.path.join(dst, modal, split))
            log[f"FlyingThings3D/{modal}/{split}"] = (l, c)

        # Monkaa scenes -> TRAIN
        mk = os.path.join(src, "Monkaa", modal)
        l, c = link_children(mk, os.path.join(dst, modal, "TRAIN"))
        log[f"Monkaa/{modal}->TRAIN"] = (l, c)

        # Driving focallength dirs -> TRAIN
        dr = os.path.join(src, "Driving", modal)
        l, c = link_children(dr, os.path.join(dst, modal, "TRAIN"))
        log[f"Driving/{modal}->TRAIN"] = (l, c)
    return log


def validate(dst, gwc_repo):
    """Resolve a sample of GwcNet list paths against the flat view."""
    results = {}
    for listname in ("sceneflow_train.txt", "sceneflow_test.txt"):
        lp = os.path.join(gwc_repo, "filenames", listname)
        if not os.path.isfile(lp):
            results[listname] = "list-missing"
            continue
        with open(lp) as f:
            lines = f.read().splitlines()
        # sample up to 200 spread across the list
        step = max(1, len(lines) // 200)
        sample = lines[::step]
        ok = miss = 0
        first_miss = None
        for ln in sample:
            parts = ln.split()
            if len(parts) < 3:
                continue
            allok = True
            for p in parts[:3]:
                if not os.path.exists(os.path.join(dst, p)):
                    allok = False
                    if first_miss is None:
                        first_miss = p
                    break
            ok, miss = (ok + 1, miss) if allok else (ok, miss + 1)
        results[listname] = (ok, miss, len(lines), first_miss)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.environ.get("VDFNET_DATA", "data/SceneFlow"))
    ap.add_argument("--dst", default=os.environ.get("VDFNET_DATA", "data/SceneFlow") + "_flat")
    ap.add_argument("--gwc", default=os.path.join(os.environ.get("WORKSPACE", "."), "GwcNet_base"),
                    help="GwcNet repo to validate filename lists against")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        sys.exit(f"[ERR] src not found: {args.src}")

    print(f"[flat] building symlink view: {args.dst}  <-  {args.src}")
    log = build(args.src, args.dst)
    print("[flat] link summary (linked, conflicts):")
    for k, (l, c) in log.items():
        flag = "  <-- CONFLICTS" if c else ""
        print(f"   {k:42s} linked={l:5d} conflict={c}{flag}")

    print("\n[flat] top-level of flat view:")
    for modal in ("frames_finalpass", "disparity"):
        for split in ("TRAIN", "TEST"):
            p = os.path.join(args.dst, modal, split)
            n = len(os.listdir(p)) if os.path.isdir(p) else 0
            print(f"   {modal}/{split}: {n} entries")

    if args.gwc and os.path.isdir(args.gwc):
        print(f"\n[flat] validating against {args.gwc}/filenames/ ...")
        res = validate(args.dst, args.gwc)
        all_good = True
        for name, r in res.items():
            if r == "list-missing":
                print(f"   {name}: LIST MISSING"); all_good = False; continue
            ok, miss, total, fm = r
            print(f"   {name}: sample {ok} OK / {miss} MISSING (of {total} lines)"
                  + (f"  first-missing={fm}" if fm else ""))
            if miss:
                all_good = False
        print("-" * 56)
        if all_good:
            print(f"[PASS] flat view resolves GwcNet lists. Use:")
            print(f"       --datapath {args.dst}")
        else:
            print("[WARN] some paths unresolved — paste this output back, "
                  "do NOT start training yet.")
    else:
        print(f"\n[flat] (no --gwc repo to validate; flat view built at {args.dst})")


if __name__ == "__main__":
    main()
